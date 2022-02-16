from models.experimental import attempt_load
from utils.general import colorstr, check_img_size, check_requirements, file_size, set_logging

from models.common import Conv
from utils.activations import Hardswish, SiLU
import torch
import torch.nn as nn

def export(
    weights='./yolov5s.pt',  # weights path
    device='cpu',   # cuda device, i.e. 0 or 0,1,2,3 or cpu
    
    ):


    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.eval()
    img_size = (640, 640)
    batch_size = 1
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
    img = torch.zeros(batch_size, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection
    img = torch.zeros(1, 3, 640, 640).to(device)  # image size(1,3,320,192) iDetection
    # TorchScript export ----------------------------------------------------------------------------------------------
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    print(img.shape)
    res = model(img)
    print(res[0].shape)
    print(res)
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

if __name__ == "__main__":
    export()