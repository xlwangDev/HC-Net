import albumentations as A
import cv2

# https://albumentations.ai/docs/getting_started/keypoints_augmentation/
def train_transform(rotation_range=45):
    return A.Compose([
        A.Perspective(pad_mode = cv2.BORDER_CONSTANT, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=rotation_range, interpolation=1,\
             border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, always_apply=False, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
    
def kitti_transform(rotation_range=10):
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        ], 
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

if __name__ == '__main__':
    transform = train_transform()
    keypoints = [(280, 203),]
    # transformed = transform(image=img1, keypoints=keypoints)
    # transformed_keypoints = transformed['keypoints']
    # temp = transformed["image"]
    # transformed_keypoints[0]
