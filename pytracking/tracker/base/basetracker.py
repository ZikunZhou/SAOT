from _collections import OrderedDict

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None


    def predicts_segmentation_mask(self):
        return False


    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track_vot_initialize(self, image, init_state, video_name='default'):
        self.initialize(image, {'init_bbox': init_state, 'video_name': video_name})

    def track_pysot_initialize(self, image, init_state, video_name='default'):
        self.initialize(image, {'init_bbox': init_state, 'video_name': video_name})

    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_vot_frame(self, image, gt):
        out = self.track(image, gt)
        state = out['target_bbox']
        return state

    def track_pysot_frame(self, image, gt=None):
        """the key best_score is used for long-term tracking"""
        out = self.track(image, gt)
        out['best_score'] = 1.0
        return out

    #def track_pysot_frame(self, image, gt=None, visualize_flag=False, video_name=''):
    #    """the key best_score is used for long-term tracking"""
    #    out = self.track(image, gt)#, visualize_flag, video_name)
    #    out['best_score'] = 1.0
    #    return out

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')
