import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'
import torch
import gradio as gr
from functools import partial
from huggingface_hub import snapshot_download

from freesplatter.webui.runner import FreeSplatterRunner
from freesplatter.webui.tab_img_to_3d import create_interface_img_to_3d
from freesplatter.webui.tab_views_to_3d import create_interface_views_to_3d
from freesplatter.webui.tab_views_to_scene import create_interface_views_to_scene


os.makedirs('./ckpts/Hunyuan3D-1', exist_ok=True)
snapshot_download('tencent/Hunyuan3D-1', repo_type='model', local_dir='./ckpts/Hunyuan3D-1')

torch.set_grad_enabled(False)
device = torch.device('cuda')
runner = FreeSplatterRunner(device)


_HEADER_ = '''
# FreeSplatter 🤗 Gradio Demo
\n\nOfficial demo of the paper [FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction](https://arxiv.org/abs/2404.07191). [[Github]](https://github.com/TencentARC/FreeSplatter)  
**FreeSplatter** is a feed-forward framework capable of generating high-quality 3D Gaussians from **uncalibrated** sparse-view images and recovering their camera parameters in mere seconds.  
'''

_IMG_TO_3D_HELP_ = '''
💡💡💡**Usage Tips:**
- This demo supports various multi-view diffusion models, including [Hunyuan3D](https://github.com/Tencent/Hunyuan3D-1) Std and [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) v1.1/v1.2. You can try different models to get the best result.
- Try clicking the \U0001f3b2\ufe0f button to use a different `Random seed` (default: 42) for diverse outputs.
- In most cases, using `2DGS` leads to better mesh geometry than `3DGS`. Please refer to the [2DGS paper](https://arxiv.org/abs/2403.17888).
- You can adjust the views used for reconstruction to alleviate the blurry texture problem caused by multi-view inconsistency.
'''

_CITE_ = r"""
If FreeSplatter is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/FreeSplatter' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/FreeSplatter?style=social)](https://github.com/TencentARC/FreeSplatter)
---
📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{xu2024freesplatter,
  title={FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction},
  author={Xu, Jiale and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint},
  year={2024}
}
```
📋 **License**
Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/spaces/TencentARC/FreeSplatter/blob/main/LICENSE) for details.
📧 **Contact**
If you have any questions, feel free to open a discussion or contact us at <b>bluestyle928@gmail.com</b>.
"""


with gr.Blocks(analytics_enabled=False, title='FreeSplatter Demo', theme=gr.themes.Ocean()) as demo:
    gr.Markdown(_HEADER_)

    with gr.Tabs() as main_tabs:
        with gr.TabItem('Image-to-3D', id='tab_img_to_3d'):
            gr.Markdown(_IMG_TO_3D_HELP_)

            with gr.Tabs() as sub_tabs_img_to_3d:
                with gr.TabItem('Hunyuan3D Std', id='tab_hunyuan3d_std'):
                    _, var_img_to_3d_hunyuan3d_std = create_interface_img_to_3d(
                        runner.run_segmentation,
                        runner.run_img_to_3d, 
                        model='Hunyuan3D Std')
                with gr.TabItem('Zero123++ v1.1', id='tab_zero123plus_v11'):
                    _, var_img_to_3d_zero123plus_v11 = create_interface_img_to_3d(
                        runner.run_segmentation,
                        runner.run_img_to_3d, 
                        model='Zero123++ v1.1')
                with gr.TabItem('Zero123++ v1.2', id='tab_zero123plus_v12'):
                    _, var_img_to_3d_zero123plus_v12 = create_interface_img_to_3d(
                        runner.run_segmentation,
                        runner.run_img_to_3d, 
                        model='Zero123++ v1.2')

    gr.Markdown(_CITE_)

    demo.launch()
