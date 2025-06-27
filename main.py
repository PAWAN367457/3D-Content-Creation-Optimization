import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
# torch.cuda.set_per_process_memory_fraction(0.8)  
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *
from DPT.dpt.models import DPTDepthModel
import torchvision.transforms as T
from scipy.ndimage import median_filter
import DPT.util.io
# BLIP
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

from PIL import Image
import clip
import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)


# torch.autograd.set_detect_anomaly(True)

def render_3d_mesh(obj_path, image_size=256, num_views=50):
    """
    Render a 3D mesh from multiple viewpoints
    
    Args:
        obj_path: Path to the .obj file
        image_size: Size of the rendered images
        num_views: Number of viewpoints to render from
    
    Returns:
        List of rendered images as numpy arrays
    """
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the .obj file
    verts, faces, aux = load_obj(obj_path)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)
    
    # Create a Meshes object
    # Initialize each vertex to be white in color
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(
        verts=[verts],
        faces=[faces_idx],
        textures=textures
    )
    
    # Create a rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )
    
    # Generate camera views around the mesh
    rendered_images = []
    for i in range(num_views):
        # Calculate angle for this view
        angle = 360.0 * i / num_views
        
        # Create a camera
        R, T = look_at_view_transform(dist=3.0, elev=0, azim=angle)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        
        # Create lights
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        
        # Create a renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        
        # Render the mesh
        images = renderer(mesh)
        
        # Convert to numpy and append to list
        rgb = images[0, ..., :3].cpu().numpy()
        rendered_images.append(rgb)
    
    return rendered_images

import torch
import clip
from PIL import Image
import numpy as np

def evaluate_clip_alignment(rendered_images, text_prompt):
    """
    Evaluate how well rendered images align with a text prompt using CLIP
    
    Args:
        rendered_images: List of rendered images as numpy arrays
        text_prompt: Text prompt to compare against
    
    Returns:
        Average similarity score
    """
    # Load CLIP model
    content = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    content.append("\nIndividual scores:")
    
    
    # Preprocess and encode text
    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Process each image and calculate similarity
    similarities = []
    for img in rendered_images:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        
        # Preprocess and encode image
        image = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * text_features @ image_features.T).item()
        content.append(f'{ similarity} ')
        # content.append(similarity)
        similarities.append(similarity)
    
    # Return average similarity
    content.append("\n")
    content.append("\nAvarage scores:")
    content.append(f'{sum(similarities) / len(similarities)}')
    return "\n".join(content)
import os
from PIL import Image
import numpy as np

def export_rendered_images(rendered_images, output_dir):
    """
    Export rendered images as PNG files
    
    Args:
        rendered_images: List of rendered images as numpy arrays
        output_dir: Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each image
    for i, img in enumerate(rendered_images):
        # Convert from float [0,1] to uint8 [0,255]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        
        # Convert to PIL Image and save
        pil_img = Image.fromarray(img)
        filename = os.path.join(output_dir, f"rendered_image_{i+1}.png")
        pil_img.save(filename)
    
    print(f"Exported {len(rendered_images)} images to {output_dir}")

import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def compute_aspect_ratios(faces, vertices):
    aspect_ratios = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-12))  # avoid sqrt of negative
        if area == 0:
            aspect_ratios.append(0)
            continue
        aspect = max(a, b, c) / (2 * area / s)  # longest edge over triangle height
        aspect_ratios.append(aspect)
    return np.array(aspect_ratios)

def write_to_file(output_file, content):
    with open(output_file, 'w') as file:
        file.write(content)

def analyze_mesh(obj_path):
    mesh = trimesh.load(obj_path)
    
    content = []
    content.append("\n===== Mesh Summary =====")
    content.append(str(mesh))
    content.append(f"Is watertight: {mesh.is_watertight}")
    content.append(f"Euler number: {mesh.euler_number}")
    content.append(f"Number of vertices: {len(mesh.vertices)}")
    content.append(f"Number of faces: {len(mesh.faces)}")
    content.append(f"Surface area: {mesh.area:.2f}")
    content.append(f"Volume: {mesh.volume:.2f}")

    content.append("\n===== Face Quality Metrics =====")
    aspect_ratios = compute_aspect_ratios(mesh.faces, mesh.vertices)
    content.append(f"Min aspect ratio: {aspect_ratios.min():.2f}")
    content.append(f"Max aspect ratio: {aspect_ratios.max():.2f}")
    content.append(f"Mean aspect ratio: {aspect_ratios.mean():.2f}")

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        content.append("\n===== UV Map Info =====")
        uv = mesh.visual.uv
        content.append(f"UV bounds: U({uv[:,0].min():.2f}, {uv[:,0].max():.2f}), V({uv[:,1].min():.2f}, {uv[:,1].max():.2f})")
    else:
        content.append("\nNo UV map found.")
    
    return "\n".join(content)

def analyze_texture(texture_path):
    content = ["\n===== Albedo Texture Info ====="]
    if os.path.exists(texture_path):
        img = Image.open(texture_path)
        content.append(f"Image format: {img.format}")
        content.append(f"Texture size: {img.size[0]} x {img.size[1]} (Width x Height)")
        content.append(f"Mode: {img.mode}")
    else:
        content.append(f"Texture not found at {texture_path}")
    
    return "\n".join(content)

def analyze_topology(mesh):
    content = []
    content.append("\n===== Mesh Topology Analysis =====")

    non_manifold_edges = mesh.edges_unique[mesh.edges_unique_length != 2]
    content.append(f"Non-manifold edges: {len(non_manifold_edges)}")

    face_areas = mesh.area_faces
    degenerate_faces = np.sum(face_areas < 1e-10)
    content.append(f"Degenerate faces (zero/small area): {degenerate_faces}")

    aspect_ratios = compute_aspect_ratios(mesh.faces, mesh.vertices)
    thin_triangles = np.sum(aspect_ratios > 10)
    content.append(f"Long thin triangles (aspect ratio > 10): {thin_triangles}")

    if mesh.volume > 0:
        compactness = mesh.volume ** 2 / (mesh.area ** 3)
        content.append(f"Compactness score: {compactness:.6f}")
    else:
        content.append("Compactness not computable (volume <= 0)")

    return "\n".join(content)

def run_analysis(obj_file, texture_file):
    mesh = trimesh.load(obj_file)

    content = []
    content.append(analyze_mesh(obj_file))
    content.append(analyze_topology(mesh))
    content.append(analyze_texture(texture_file))

    # Determine the directory of the .obj file and create output path
    output_directory = os.path.dirname(obj_file)
    output_file = os.path.join(output_directory, "mesh_analysis_results.txt")

    # Write all results to the output file
    # write_to_file(output_file, "\n".join(content))
    return "\n".join(content)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--final', action='store_true', help="final train mode")
    parser.add_argument('--refine', action='store_true', help="refine mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--depth_model', type=str, default='dpt_hybrid', help='choose from [dpt_large, dpt_hybrid]')
    parser.add_argument('--guidance_scale', type=float, default=10)
    parser.add_argument('--need_back', action='store_true', help="use back text prompt")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")

    parser.add_argument('--ref_path', default=None, type=str, help="use image as referance, only support alpha image")


    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--refine_iters', type=int, default=3000, help="refine iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=2048, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--diff_iters', type=int, default=400, help="training iters that only use albedo shading")
    parser.add_argument('--step_range', type=float, nargs='*', default=[0.2, 0.6])
    
    # model options
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.1, help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='tcnn', choices=['grid', 'tcnn', 'sdf', 'vanilla', 'normal'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=128, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=128, help="render height for NeRF in training")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fov', type=float, default=20, help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[15, 25], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[70, 110], help="training camera phi range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[0, 360], help="training camera phi range")
    
    parser.add_argument('--lambda_entropy', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=1e-3, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=1, help="loss scale for surface smoothness")
    parser.add_argument('--lambda_img', type=float, default=1e3, help="loss scale for ref loss")
    parser.add_argument('--lambda_depth', type=float, default=1, help="loss scale for depth loss")
    parser.add_argument('--lambda_clip', type=float, default=1, help="loss scale for clip loss")
    
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    parser.add_argument('--max_depth', type=float, default=10.0, help="farthest depth")
    
    opt = parser.parse_args()
    opt.cuda_ray = True
    optDict = opt.__dict__
    opt.workspace = os.path.join('results', opt.workspace)
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True) 
    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)
    seed_everything(opt.seed)

    # load depth network
    net_w = net_h = 384
    depth_model = DPTDepthModel(
        path="dpt_weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_transform = T.Compose(
    [
        T.Resize((384, 384)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_model.to(device)

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        warm_up_with_cosine_lr = lambda iter: iter / opt.warm_iters if iter <= opt.warm_iters \
            else max(0.5 * ( math.cos((iter - opt.warm_iters) /(opt.iters - opt.warm_iters) * math.pi) + 1), 
                        opt.min_lr / opt.lr)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    if opt.guidance == 'stable-diffusion':
        from nerf.sd import StableDiffusion
        guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, step_range=opt.step_range)
    elif opt.guidance == 'clip':
        from nerf.clip import CLIP
        guidance = CLIP(device)
    else:
        raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

    ref_imgs = cv2.imread(opt.ref_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
    image_pil = Image.open(opt.ref_path).convert("RGB")

    # generated caption
    if opt.text == None:
        print("load blip2 for image caption...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32).to("cuda")
        inputs = processor(image_pil, return_tensors="pt").to("cuda", torch.float32)
        out = blip_model.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        caption = caption.replace("there is ", "")
        caption = caption.replace("close up", "photo")
        for d in ["black background", "white background"]:
            if d in caption:
                caption = caption.replace(d, "ground")
        print("Caption: ", caption)
        opt.text = caption

    with open(os.path.join(opt.workspace, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


    # only support alpha photo input.
    imgs = cv2.cvtColor(ref_imgs, cv2.COLOR_BGRA2RGBA)
    imgs = cv2.resize(imgs, (512, 512), interpolation=cv2.INTER_AREA)
    ref_imgs = (torch.from_numpy(imgs)/255.).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    ori_imgs = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + (1 - ref_imgs[:, 3:, :, :])
    
    mask = imgs[:, :, 3:]
    # mask[mask < 0.5 * 255] = 0
    # mask[mask >= 0.5 * 255] = 1 
    kernel = np.ones(((5,5)), np.uint8) ##11
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = (mask == 0)
    mask = (torch.from_numpy(mask)).unsqueeze(0).unsqueeze(0).to(device)
    depth_mask = mask
    
    # depth estimation
    with torch.no_grad():
        depth_prediction = depth_model.forward(depth_transform(ori_imgs))
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=512,
            mode="bicubic",
            align_corners=True,
        ) # [1, 1, 512, 512] [80~150]
        DPT.util.io.write_depth_name(os.path.join(opt.workspace, "hel"  + '_depth'), depth_prediction.squeeze().cpu().numpy(), bits=2)
        disparity = imageio.imread(os.path.join(opt.workspace, "hel" + '_depth.png')) / 65535.
        disparity = median_filter(disparity, size=5)
        depth = 1. / np.maximum(disparity, 1e-2)
    
    depth_prediction = torch.tensor(depth, device=device)
    depth_mask = torch.tensor(depth_mask, device=device)
    # normalize estimated depth
    depth_prediction = depth_prediction * (~depth_mask) + torch.ones_like(depth_prediction) * (depth_mask)
    depth_prediction = ((depth_prediction - 1.0) / (depth_prediction.max() - 1.0)) * 0.9 + 0.1
    # save_image(ori_imgs, os.path.join(opt.workspace, opt.workspace + '_ref.png'))

    model = NeRFNetwork(opt)
    trainer = Trainer('df', opt, model, depth_model, guidance, 
                        ref_imgs=ref_imgs, ref_depth=depth_prediction, 
                        ref_mask=depth_mask, ori_imgs=ori_imgs, 
                        device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
    

    if opt.test:
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=33).dataloader()
        trainer.test(test_loader, write_video=True)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
            
    else:
        
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=50).dataloader()
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        if opt.final:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
            trainer.test(test_loader, write_image=False, write_video=True)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
            
        if opt.refine:
            mv_loader = NeRFDataset(opt, device=device, type='gen_mv', H=opt.H, W=opt.W, size=33).dataloader()
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
            trainer.test(mv_loader, save_path=os.path.join(opt.workspace, 'mvimg'), write_image=True, write_video=False)
            trainer.refine(os.path.join(opt.workspace, 'mvimg'), opt.refine_iters, test_loader)

    obj_path = "/home/nitin/Make-It-3D/"+opt.workspace +"/mesh/mesh.obj"
    text_prompt = os.path.join(opt.workspace, opt.text.replace(" ", "_") + '_depth')
    
    # Render the mesh from multiple viewpoints
    rendered_images = render_3d_mesh(obj_path)
    # Specify where you want to save the images
    output_dir = "/home/nitin/Make-It-3D/results/"+opt.workspace+"/rendered_images/"

    # Export the images
    export_rendered_images(rendered_images, output_dir)
    # Evaluate alignment with CLIP
    content = []
    # avg_score, individual_scores = evaluate_clip_alignment(rendered_images, text_prompt)

    # text_arr = [str(x) for x in individual_scores]
    # text = str(avg_score)
    # content.append(text)
    # content.append(text_arr)
    output_directory = os.path.dirname(obj_path)
    output_file = os.path.join(output_directory, "mesh_analysis_results.txt")

    # Write all results to the output file
    # write_to_file(output_file, "\n".join(content))
    # print(f"Average CLIP alignment score: {avg_score:.2f}")
    # print(f"Individual scores: {[f'{score:.2f}' for score in individual_scores]}")

    obj_file = obj_path  # Replace with your actual .obj file path
    texture_file = "/home/nitin/Make-It-3D/"+opt.workspace+"/mesh/albedo.png"  # Replace with your actual texture file path
    content.append(run_analysis(obj_file, texture_file))


    content.append(evaluate_clip_alignment(rendered_images, text_prompt))
    write_to_file(output_file, "\n".join(content))
    print(f"Analysis results saved to: {output_file}")
    
        