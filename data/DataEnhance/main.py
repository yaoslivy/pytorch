from PIL import Image
from torchvision import transforms
from IPython.display import display

if __name__ == "__main__":
    print("**convert operation**")
    img = Image.open('jk.jpg')
    display(img)
    print(type(img))
    # PIL.Image -> torch.Tensor
    img1 = transforms.ToTensor()(img)
    print(type(img1))
    # torch.Tensor -> PIL.Image 
    img2 = transforms.ToPILImage()(img1)
    print(type(img2))

    print("**resize operation**")
    # Resize operation
    resize_img_oper = transforms.Resize((200, 200))
    # original img
    orig_img = Image.open('jk.jpg')
    display(orig_img)
    #Resize 
    img = resize_img_oper(orig_img)
    display(img)

    print("**crop operation**")
    # define crop operation
    center_crop_oper = transforms.CenterCrop((60, 70))
    random_crop_oper = transforms.RandomCrop((80, 80))
    five_crop_oper = transforms.FiveCrop((60, 70))
    # original img
    orig_img = Image.open("jk.jpg")
    display(orig_img)
    # center crop
    img1 = center_crop_oper(orig_img)
    display(img1)
    # random crop
    img2 = random_crop_oper(orig_img)
    display(img2)
    # four corners and the middle crop
    imgs = five_crop_oper(orig_img)
    for img in imgs:
        display(img)

    print("**flip operation**")
    display(orig_img)
    # define flip operation
    h_flip_oper = transforms.RandomHorizontalFlip(p=1)
    v_flip_oper = transforms.RandomVerticalFlip(p=1)
    # original img
    orig_img = Image.open('jk.jpg')
    display(orig_img)
    # horizon flip
    img1 = h_flip_oper(orig_img)
    display(img1)
    # vertical flip
    img2 = v_flip_oper(orig_img)
    display(img2)

    print("**normalize operation**")
    # define normalize operation
    norm_oper = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # original img
    orig_img = Image.open('jk.jpg')
    display(orig_img)
    # img -> Tensor
    img_tensor = transforms.ToTensor()(orig_img)
    # normalize
    tensor_norm = norm_oper(img_tensor)
    # tensor -> img
    img_norm = transforms.ToPILImage()(tensor_norm)
    display(img_norm)

    print("**compose operation**")
    #define compose operation
    composed = transforms.Compose([transforms.Resize((200, 200)),
                                  transforms.RandomCrop(80)]) 
    # original img
    orig_img = Image.open('jk.jpg')
    display(orig_img)
    img = composed(orig_img)
    display(img)







