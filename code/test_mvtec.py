import os
import cv2
from metrics import cal_pro_score
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
# from visualization import visualizer

parser = argparse.ArgumentParser("KAG_prompt", add_help=True)
# paths
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=195) #1-shot:195, 2-shot:195, 4-shot:194

command_args = parser.parse_args()

describles = {
    'bottle': 'bottle',
    'cable': 'cable',
    'capsule': 'capsule',
    'carpet': 'carpet',
    'grid': 'grid',
    'hazelnut': 'hazelnut',
    'leather': 'leather',
    'metal_nut': 'metal nut',
    'pill': 'pill',
    'screw': 'screw',
    'tile': 'tile',
    'toothbrush': 'toothbrush',
    'transistor': 'transistor',
    'wood': 'wood',
    'zipper': 'zipper'
}

FEW_SHOT = command_args.few_shot


# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'anomalygpt_ckpt_path': './ckpt/train_visa/train_on_visa.pt',
    'stage': 2,
    'features_list': [6, 12, 18, 24],
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda().eval()

p_auc_list = []
i_auc_list = []
p_pro_list =[]
ap_list = []
def predict(
        input,
        image_path,
        normal_img_path,
        r,
):
    prompt_text = input
    pixel_output, cls_score = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'r': r
    })

    return pixel_output, cls_score


root_dir = '/mnt/sda/fenfangtao/MVTecAD'

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

precision = []
r = 0.1
for c_name in CLASS_NAMES:
    base_path = "/mnt/sda/fenfangtao/MVTecAD/" + c_name + "/train/good/"
    normal_img_paths = []
    for i in range(command_args.k_shot):
        round_number = command_args.round + i
        file_path = base_path + str(round_number).zfill(3) + ".png"

        if not Path(file_path).is_file():
            break

        normal_img_paths.append(file_path)

    if not normal_img_paths:
        all_image_paths = sorted(Path(base_path).glob("*.png"))
        normal_img_paths = all_image_paths[-command_args.k_shot:]

    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "test" in file_path and 'png' in file and c_name in file_path:
                if FEW_SHOT:
                    anomaly_map, score = predict(describles[c_name], file_path, normal_img_paths, r)
                else:
                    anomaly_map, score = predict(describles[c_name], file_path, [], r)

                is_normal = 'good' in file_path.split('/')[-2]

                if is_normal:
                    img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
                else:
                    mask_path = file_path.replace('test', 'ground_truth')
                    mask_path = mask_path.replace('.png', '_mask.png')
                    img_mask = Image.open(mask_path).convert('L')

                img_mask = mask_transform(img_mask)
                img_mask[img_mask > 0.1], img_mask[img_mask <= 0.1] = 1, 0
                img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()

                anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

                # save_path = '/mnt/sda/fenfangtao/AnomalyGPT-main/code/visualize/'
                # visualizer(file_path, anomaly_map, 224, save_path, c_name, 'mvtec')

                p_label.append(img_mask)
                p_pred.append(anomaly_map)

                i_label.append(1 if not is_normal else 0)

                # i_pred.append(anomaly_map.max())
                k = 30
                flat_matrix = anomaly_map.ravel()
                top_k_indices = np.argpartition(-flat_matrix, k)[:k]
                top_k_values = flat_matrix[top_k_indices]
                score1 = np.mean(top_k_values)
                i_pred.append(0.1 * score.cpu().detach().numpy() + 0.9 * score1)

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
    ap = round(average_precision_score(i_label, i_pred) * 100, 2)

    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    p_pro_list.append(p_pro)
    ap_list.append(ap)

    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)
    print(c_name, "p_pro:", p_pro)
    print(c_name, "ap:", ap)

print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("p_PRO:", torch.tensor(p_pro_list).mean())
print("ap:", torch.tensor(ap_list).mean())
