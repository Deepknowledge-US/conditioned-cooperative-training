from icevision.all import *
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.progress import CSVLogger
from icevision.engines.fastai.adapters import FastaiMetricAdapter
from icevision.models.mmdet.fastai.callbacks import _ModelWrap
from fastprogress import progress_bar
import glob
import json
from copy import copy
import plotly.express as px
import random
import argparse
import logging
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BS = 2
image_size = 800
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=image_size+128), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
unsupervised_tfms = tfms.A.Adapter([tfms.A.Normalize()])

# Architecture
model_type = models.mmdet.deformable_detr
backbone = model_type.backbones.r50_16x2_50e_coco(pretrained=True)


class UnsupervisedParser(Parser):
    def __init__(self, template_record, images_dir):
        super().__init__(template_record=template_record)
        self.images_dir = images_dir
        self.images = glob.glob(str(images_dir/'*.jpg'))
        self.class_map = ClassMap(['gun'])

    def __iter__(self) -> Any:
        for i in self.images:
            yield i

    def __len__(self) -> int:
        return len(self.images)

    def record_id(self, o) -> Hashable:
        return o

    def parse_fields(self, o, record, is_new):
        if is_new:
            filepath = self.images_dir / 'images' / o
            w, h = PIL.Image.open(filepath).size
            record.set_filepath(filepath)
            record.set_img_size(ImgSize(width=w, height=h))
            record.detection.set_class_map(self.class_map)


def read_datasets(supervised_dir, unsupervised_dir):
    with open(f'{supervised_dir}/ann_ids/train.txt', 'r') as f:
        train_ids = f.read().splitlines()
    with open(f'{supervised_dir}/ann_ids/control.txt', 'r') as f:
        control_ids = f.read().splitlines()
    with open(f'{supervised_dir}/ann_ids/test.txt', 'r') as f:
        test_ids = f.read().splitlines()

    parser = parsers.VOCBBoxParser(annotations_dir=supervised_dir/'annotations', images_dir=supervised_dir/'images')

    data_splitter = FixedSplitter([train_ids, test_ids])
    train_records, valid_records = parser.parse(data_splitter, autofix=True)
    _, control_records = parser.parse(FixedSplitter([train_ids, control_ids]), autofix=True)

    template_record = ObjectDetectionRecord()
    unsupervised_parser = UnsupervisedParser(template_record, images_dir=unsupervised_dir/'images')
    unsupervised_records = unsupervised_parser.parse(SingleSplitSplitter(), autofix=True)[0]

    # Datasets
#     train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)
    control_ds = Dataset(control_records, valid_tfms)
    sample_of_unsupervised_ds = Dataset(unsupervised_records[:500], tfm=unsupervised_tfms)

    return train_records, valid_ds, control_ds, unsupervised_records, sample_of_unsupervised_ds, parser.class_map


class DistillCB(fastai.Callback):
    def after_step(self):
        if not self.model.training: return
        if selfdistill.supervised: return
        selfdistill.ema_from_student_to_teacher()


class ThresholdFinder(fastai.Callback):
    def before_fit(self):
        self.state_dict_copy = self.state_dict()
        selfdistill.cocometrics = {}
        self.threshold = 0.5
        n_images = selfdistill.set_unsupervised_dl_with_threshold(self.threshold, is_sample=True)
        selfdistill.cocometrics[self.threshold] = []
        selfdistill.copy_weights_from_teacher_to_student()
        
    def before_epoch(self):
        if not self.model.training: return
        if self.threshold in selfdistill.cocometrics and len(selfdistill.cocometrics[self.threshold]) == selfdistill.threshold_epochs:
#             print(f'Results of threshold {self.threshold}:', selfdistill.cocometrics[self.threshold])
            increment = 0.05 if selfdistill.threshold_highprecision else 0.1
            self.threshold = np.round(self.threshold + increment, decimals=2)
            if self.threshold>0.89: raise fastai.CancelFitException
            n_images = selfdistill.set_unsupervised_dl_with_threshold(self.threshold, is_sample=True)
            if n_images < 1000: raise fastai.CancelFitException
            selfdistill.cocometrics[self.threshold] = []
            selfdistill.copy_weights_from_teacher_to_student()
            self.opt.clear_state()

    def after_epoch(self):
        if not self.model.training: return
        selfdistill.cocometrics[self.threshold].append(self.recorder.log[3])
        
@torch.no_grad()
def _predict_from_dl_with_threshold(
    predict_fn,
    model: nn.Module,
    infer_dl: DataLoader,
    threshold: float,
    n_images: int = None,
    keep_images: bool = False,
    show_pbar: bool = True,
    **predict_kwargs,
) -> List[Prediction]:
    all_preds = []
    stop_batch = int(1 * len(infer_dl.dataset)/BS)
    
    batch_idx = 0
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(
            model=model,
            batch=batch,
            records=records,
            keep_images=keep_images,
            **predict_kwargs,
        )
        for br, pred in zip(records, preds):
#            bboxes = list(filter(lambda b: (b.ymax - b.ymin) < 250, pred.pred.detection.bboxes))
#            lower_than_size = len(pred.pred.detection.bboxes) == len(bboxes) 
#            if lower_than_size and all(pred.pred.detection.scores > threshold):
            if all(pred.pred.detection.scores > threshold) and br.detection.record_id not in selfdistill.used_unsupervised_images:
                br.detection.set_bboxes(pred.pred.detection.bboxes)
                br.detection.set_labels(pred.pred.detection.labels)
                selfdistill.used_unsupervised_images.append(br.detection.record_id)
            else:
                br.detection.set_bboxes([])
                br.detection.set_labels([])
        cleaned_records = list(filter(lambda r: len(r.detection.bboxes) > 0, records))        
        all_preds.extend(cleaned_records)
#         all_preds.extend(preds)
        if n_images != None and len(all_preds) >= n_images:
            return all_preds
        elif n_images != None and batch_idx >= stop_batch:
            return all_preds
        else:
            batch_idx += 1
    return all_preds

def predict_from_dl_with_threshold(
    model: nn.Module,
    infer_dl: DataLoader,
    threshold: float,
    n_images: int = None,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl_with_threshold(
        predict_fn=model_type.prediction._predict_batch,
        model=model,
        infer_dl=infer_dl,
        threshold=threshold,
        n_images=n_images,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )

def sum_up_cocometrics(cocometrics):
    resumed_cocometrics = {}
    for k, v in cocometrics.items():
        new_k = np.floor(k*10)/10
        if new_k in resumed_cocometrics:
            resumed_cocometrics[new_k].extend(v)
        else:
            resumed_cocometrics[new_k] = v
    return resumed_cocometrics


def plot_cocometrics(cocometrics):
    thresholds = []
    results = []
    for k, v in cocometrics.items():
        thresholds.append(k)
        results.append(np.median(cocometrics[k]))
    fig = px.bar(x=thresholds, y=results, range_y=[min(results)-0.001,
                                                   max(results)+0.001],
                 title="Median mAP per threshold range",
                 labels={ "x": "Threshold range", "y": "Median mAP" })
    fig.show()

def get_best_threshold():
    selfdistill.find_threshold()
    mean_metric = {k: np.median(v) for k,v in selfdistill.cocometrics.items()}
    return max(mean_metric, key=mean_metric.get)


class SelfDistill():
    def __init__(self, model_type, backbone, num_classes, train_records, valid_dl, control_dl, unsupervised_records,
                 sample_of_unsupervised_ds, m=0.999, n_unsupervised_images=30000):
        self.model_type = model_type
        self.unsupervised_records = unsupervised_records
        self.sample_of_unsupervised_ds = sample_of_unsupervised_ds
#         self.dls = [train_dl, valid_dl]
        self.train_records = train_records
        self.valid_dl = valid_dl
        self.set_dls(0, split_size=len(train_records))
        self.control_dl = control_dl
        self.metrics = [COCOMetric(metric_type=COCOMetricType.bbox, print_summary=False)]
        self.m = m
        self.n_unsupervised_images = n_unsupervised_images
        self.set_random_unsupervised_set()
        # Instantiate the model
        self.teacher_model = model_type.model(backbone=backbone(pretrained=True), num_classes=num_classes)
        self.student_model = model_type.model(backbone=backbone(pretrained=True), num_classes=num_classes)
        self.teacher_learner = self.create_learner(self.teacher_model)
        self.student_learner = self.create_learner(self.student_model)
        self.learner = self.teacher_learner
        self.switch_to_teacher()
        self.used_unsupervised_images = []
    
    def reset_states(self):
        self.teacher_model = model_type.model(backbone=backbone(pretrained=True), num_classes=num_classes)
        self.student_model = model_type.model(backbone=backbone(pretrained=True), num_classes=num_classes)
        self.teacher_learner = self.create_learner(self.teacher_model)
        self.student_learner = self.create_learner(self.student_model)
        self.learner = self.teacher_learner
        self.switch_to_teacher()
    
    def set_dls(self, idx, split_size=300):
        split_size = len(self.train_records)
        split_idx = idx%int(len(self.train_records)/split_size)
        train_ds = Dataset(self.train_records[split_size*split_idx:split_size*(split_idx+1)], train_tfms)
        train_dl = model_type.train_dl(train_ds, batch_size=BS, num_workers=16, shuffle=True)
        self.dls = [train_dl, self.valid_dl]
    
    def create_learner(self, model, cbs=[DistillCB(), CSVLogger]):
        learner = self.model_type.fastai.learner(dls=self.dls, model=model, metrics=self.metrics, cbs=cbs)
        learner.m = self.m
        return learner

    def metrics_should_print_summary(self, learner, print_summary):     
        learner.metrics[0] = FastaiMetricAdapter(COCOMetric(metric_type=COCOMetricType.bbox, print_summary=print_summary))
    
    def copy_weights_from_teacher_to_student(self):
        for param_teacher, param_student in zip(self.teacher_learner.parameters(), self.student_learner.parameters()):
            param_student.data.copy_(param_teacher.data)

    def ema_from_student_to_teacher(self):
        for param_teacher, param_student in zip(self.teacher_learner.parameters(), self.student_learner.parameters()):
            param_teacher.data = param_teacher.data * self.m + param_student.data * (1. - self.m)
    
    def set_random_unsupervised_set(self):
        self.selected_records = random.sample(list(self.unsupervised_records),
                                              k=len(list(self.unsupervised_records)))
    
    def switch_to_student(self, threshold, is_sample=False):
        self.copy_weights_from_teacher_to_student()
        self.learner = self.student_learner
        self.supervised = False
        return self.set_unsupervised_dl_with_threshold(threshold, is_sample)
    
    def set_unsupervised_dl_with_threshold(self, threshold, is_sample=False):
        n_sample_images = 5000
        self.set_random_unsupervised_set()
        ds = Dataset(self.selected_records, tfm=unsupervised_tfms)
#         ds = Dataset(self.selected_records, tfm=train_tfms)
        infer_dl = model_type.infer_dl(ds, batch_size=BS, batch_tfms=None, num_workers=16)
        self.preds = predict_from_dl_with_threshold(model=self.teacher_learner.model.model, infer_dl=infer_dl,
                                                    threshold=threshold,
                                                    n_images=n_sample_images if is_sample else self.n_unsupervised_images,
                                                    keep_images=False, show_pbar=True)
        print(f'Total of images with a threshold of {threshold}: {len(self.preds[:n_sample_images if is_sample else self.n_unsupervised_images])}')
        cleaned_unsupervised_ds = Dataset(self.preds[:n_sample_images if is_sample else self.n_unsupervised_images], tfm=train_tfms)
        train_dl = model_type.train_dl(cleaned_unsupervised_ds, batch_size=BS, num_workers=16, shuffle=True)
        self.learner.dls.train = train_dl
        return len(self.preds[:n_sample_images if is_sample else None])
    
    def find_threshold(self, epochs=5, high_precision=False):
        self.threshold_epochs = epochs
        self.threshold_highprecision = high_precision
        self.student_learner = self.create_learner(self.student_model, cbs=[ThresholdFinder()])
        self.learner = self.student_learner
        self.learner.dls.valid = self.control_dl
        self.learner.fit(epochs*10 if high_precision else epochs*5, 1e-5)
        self.student_learner = self.create_learner(self.student_model)
    
    def switch_to_teacher(self):
        self.learner = self.teacher_learner
        self.supervised = True
        self.learner.dls.train = self.dls[0]
        self.learner.dls.valid = self.dls[1]
    
    def evaluate_control(self):
        results = {}
        self.teacher_learner.dls.valid = self.control_dl
        self.student_learner.dls.valid = self.control_dl
        results['teacher'] = self.teacher_learner.validate()[1]
        results['student'] = self.student_learner.validate()[1]
        self.teacher_learner.dls.valid = self.dls[1]
        self.student_learner.dls.valid = self.dls[1]
        return results
    
    def validate(self, valid_set_name='valid'):
        # Prepare learner config
        valid_set = self.control_dl if valid_set_name == 'control' else self.dls[1]
        self.teacher_learner.dls.valid = valid_set
        self.student_learner.dls.valid = valid_set
        self.metrics_should_print_summary(self.teacher_learner, True)
        self.metrics_should_print_summary(self.student_learner, True)
        
        coco_results = {'teacher': {}, 'student': {}}
        
        # Validate
        with CaptureStdout(propagate_stdout=False) as teacher_output:
            self.teacher_learner.validate()
           
        with CaptureStdout(propagate_stdout=False) as student_output:
            self.student_learner.validate()
        
        for m in teacher_output:
            if m.strip().startswith('Average'):
                coco_results['teacher'][m[1:-8]] = float(m[-5:])
        for m in student_output:
            if m.strip().startswith('Average'):
                coco_results['student'][m[1:-8]] = float(m[-5:])
        
        # Restore learner config
        self.metrics_should_print_summary(self.teacher_learner, False)
        self.metrics_should_print_summary(self.student_learner, False)
        self.teacher_learner.dls.valid = self.dls[1]
        self.student_learner.dls.valid = self.dls[1]
                
        return coco_results

    def save_log(self, output_dir):
        self.teacher_learner.csv_logger.read_log().to_csv(os.path.join(output_dir, 'log.csv'), index=False)
    
    def save_metrics(self, output_dir, verbose=True):
        coco_results = self.validate('valid')
        with open(os.path.join(output_dir, 'metrics.json'), "w") as outfile:
            json.dump(coco_results, outfile)
        if verbose:
            print(json.dumps(coco_results, indent=4))


def main():
    global selfdistill
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", help="warmup or iterative")
    parser.add_argument("--supervised_dir", required=True, help="path to supervised dataset directory")
    parser.add_argument("--unsupervised_dir", required=True, help="path to unsupervised dataset directory")
    parser.add_argument("--output", required=True, help="path folder to save model and log")
    parser.add_argument("--warmup_output", required=False, help="path folder to load warmup model")
    parser.add_argument("--lr", required=False, default='auto', help="learning rate (default: auto)")
    parser.add_argument("--epc", type=int, required=False, default=5, help="epochs per cycle")
    parser.add_argument("--attemps", type=int, required=False, default=2, help="max non-improvement attemps during iterative cooperative phase (default: 2)")
    parser.add_argument("--ct", required=False, default='auto', help="confidence threshold (default: auto)")
    parser.add_argument("--device", required=False, help="training device")
    args = parser.parse_args()
    
    if args.device:
        torch.cuda.set_device(args.device)
    
    # Set absolute output dirs
    OUTPUT_DIR = os.path.join(os.getcwd(), args.output) if not args.output.startswith('/') else args.output
    if args.warmup_output is not None:
        WARMUP_OUTPUT_DIR = os.path.join(os.getcwd(), args.warmup_output) if not args.warmup_output.startswith('/') else args.warmup_output
    elif args.warmup_output is None and args.phase == 'iterative':
        raise ValueError("--warmup_output argument necessary during iterative phase")
    
    # Create output dir and save weights, log and metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save args to output dir as a config file
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile)

    # Datasets and Dataloaders
    train_records, valid_ds, control_ds, unsupervised_records, sample_of_unsupervised_ds, class_map = read_datasets(Path(args.supervised_dir), Path(args.unsupervised_dir))
    valid_dl = model_type.valid_dl(valid_ds, batch_size=BS, num_workers=16, shuffle=False)
    control_dl = model_type.valid_dl(control_ds, batch_size=BS, num_workers=16, shuffle=False)
    num_classes = len(class_map)

    # Create methodology class
    selfdistill = SelfDistill(model_type, backbone, num_classes, train_records, valid_dl, control_dl,
                                unsupervised_records, sample_of_unsupervised_ds)

    if args.phase == 'warmup':
        # Set LR
        if args.lr == 'auto':
            LR = selfdistill.teacher_learner.lr_find().valley
            print(f'LR: {LR}')
        else:
            LR = float(args.lr)

        # Fit teacher model with supervised data
        selfdistill.teacher_learner.fine_tune(100, LR, freeze_epochs=1)
        
        # Save weights, log and metrics
        selfdistill.teacher_learner.save(os.path.join(OUTPUT_DIR, 'checkpoint'))
        selfdistill.save_log(OUTPUT_DIR)
        selfdistill.save_metrics(OUTPUT_DIR)


    elif args.phase == 'iterative':
        # Load warmup weights
        model_path = os.path.join(WARMUP_OUTPUT_DIR, 'checkpoint')
        selfdistill.teacher_learner.load(model_path)
        
        # Vars
        teacher_state_dict = selfdistill.teacher_learner.state_dict()
        teacher_state_dict_cycle = 0
        non_improvement = 0
        control_results = selfdistill.evaluate_control()
    
        # Set confidence threshold
        if args.ct == 'auto':
            confidence_threshold = get_best_threshold()
            print('BEST CONFIDENCE THRESHOLD:', confidence_threshold)
        else:
            confidence_threshold = float(args.ct)
        # Generate pseudolabels
        selfdistill.switch_to_student(confidence_threshold)
        
        # Set LR
        if args.lr == 'auto':
            LR = selfdistill.learner.lr_find().valley
            print(f'LR: {LR}')
        else:
            LR = float(args.lr)
            
        # Iterative cooperative phase
        while non_improvement < args.attemps:
            print(f'<<< CYCLE {i+1} >>>')
            
            # Fit student model with pseudolabels
            selfdistill.learner.fit_one_cycle(args.epc, LR)
            selfdistill.validate()
            
            # Check improvement over control set
            new_control_results = selfdistill.evaluate_control()
            control_diff = new_control_results['teacher'] - control_results['teacher']
            
            if control_diff < 0:  # Restore teacher if there is no improvement
                print(f"Cycle {i+1}: mAP decrease from {control_results['teacher']} to {new_control_results['teacher']} ({control_diff})")
                print(f"Restoring teacher to cycle {teacher_state_dict_cycle+1}")
                selfdistill.teacher_learner.load_state_dict(teacher_state_dict)
                non_improvement += 1

            else:  # Set new control_results, teacher_state_dict and teacher_state_dict_cycle as new best values on improvement
                print(f"Cycle {i+1}: mAP increase from {control_results['teacher']} to {new_control_results['teacher']} (+{control_diff})")
                control_results = selfdistill.evaluate_control()
                teacher_state_dict = selfdistill.teacher_learner.state_dict()
                teacher_state_dict_cycle = i
                non_improvement = 0
            
            # Create output dir and save weights, log and metrics of cycle
            cycle_output_dir = os.path.join(OUTPUT_DIR, f'cycle{i+1}')
            os.makedirs(cycle_output_dir, exist_ok=True)
            selfdistill.teacher_learner.save(os.path.join(cycle_output_dir, 'checkpoint'))
            selfdistill.save_log(cycle_output_dir)
            selfdistill.save_metrics(cycle_output_dir)
                        
            # Set confidence threshold
            if args.ct == 'auto':
                confidence_threshold = get_best_threshold()
                print('BEST CONFIDENCE THRESHOLD:', confidence_threshold)
            else:
                confidence_threshold = float(args.ct)
            # Generate pseudolabels
            selfdistill.switch_to_student(confidence_threshold)


if __name__ == "__main__":
    main()