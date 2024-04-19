import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import os
import argparse
from torchvision.utils import save_image
from torchvision.models import resnet18
from torchattacks.attack import Attack
from torch.utils.data import Dataset, DataLoader


EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 4


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PGD Attack on the surrogate clasifier."
    )
    parser.add_argument(
        "--attack_name",
        type=str,
        default="unwm_wm",
        choices=["unwm_wm", "real_wm", "wm1_wm2"],
        help="Three adversarial surrogate detector attacks tested in the paper.",
    )
    parser.add_argument(
        "--watermark_name",
        type=str,
        default="tree_ring",
        choices=["tree_ring", "stable_sig", "stegastamp"],
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2,
        choices=[2, 4, 6, 8],
        help="The perturbation radius of adversarial attacks. It will be divided by 255 in the code.",
    )
    parser.add_argument(
        "--target_label",
        type=int,
        default=0,
        choices=[0, 1],
        help="The target label for PGD targeted-attack. Labels are the ones used in surrogate model training. "
        "For umwm_wm, 0 is non-watermarked, 1 is watermarked. To remove watermarks, the target_label should be 0.",
    )

    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return parsed_args


def adv_surrogate_model_attack(x0,
    xr,
    model,
    strength=10,
):
    average_delta = None
    # Generate adversarial images
    attack = pgd_attack_classifier(
        model=model,
        eps=EPS_FACTOR * strength,
        alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
        steps=N_STEPS,
        random_start=False,
    )
    images = xr
    target_labels = torch.zeros(images.size(0), dtype=torch.long).to(xr.device)

    # PGD attack
    images_adv = attack(x0,images, target_labels, init_delta=average_delta)
    images_adv=2*images_adv-1
    return images_adv


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=None):
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.root = root
        self.transform = transform
        self.extensions = extensions

        # Load filenames from the root
        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in self.extensions
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path  # return image path to identify the image file later

    def __len__(self):
        return len(self.filenames)


class WarmupPGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x0, images, labels, init_delta=None):
        """
        Overridden.
        """
        self.model.eval()

        images = images.clone().detach().to(self.device)
        labels = labels.type(torch.FloatTensor).clone().detach().to(self.device)

        adv_images=images.clone().detach()

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        # if self.random_start:
        #     adv_images = images.clone().detach()
        #     # Starting at a uniformly random point
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(
        #         -self.eps, self.eps
        #     )
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        # elif init_delta is not None:
        #     clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
        #     adv_images = images.clone().detach() + clamped_delta
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        # else:
        #     assert False

        for i in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)
            outputs=self.model(adv_images)

            # Calculate loss
            # if self.targeted:
            #     cost = -self.loss(outputs, target_labels)
            # else:
            #     cost = self.loss(outputs, labels)

            # cost = self.loss(outputs, labels)
            cost=nn.BCELoss()(outputs[:, 0],labels)
            # cost=torch.mean(outputs)
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            # if (i==0): print(i,adv_images,outputs,cost)

            adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        # print(outputs[0],torch.mean(delta))

        return adv_images


def pgd_attack_classifier(model, eps, alpha, steps, random_start=True):
    # Create an instance of the attack
    attack = WarmupPGD(
        model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
    )

    # Set targeted mode
    # attack.set_mode_targeted_by_label(quiet=True)

    return attack