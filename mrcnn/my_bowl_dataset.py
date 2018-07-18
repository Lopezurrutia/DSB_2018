from utils import Dataset
import os
import numpy as np
import skimage.io 


class BowlDataset(Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_bowl(self, folderpaths):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("bowl", 1, "nuclei")

        #Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. 

        # Add images
        for i in range(len(folderpaths)):
            self.add_image("bowl", image_id=i, path=folderpaths[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image_path = os.path.join(image_path, 'images', '{}.png'.format(os.path.basename(image_path)))
        image = skimage.io.imread(image_path)
        image = image[:,:,:3]
        return image
        
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bowl":
            return info["bowl"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_paths = os.path.join(image_path, 'masks', '*.png')
        mask = skimage.io.imread_collection(mask_paths).concatenate()
        mask = np.rollaxis(mask,0,3)
        mask = np.clip(mask,0,1)
        class_ids = np.array([1]* mask.shape[2])
        return mask, class_ids.astype(np.int32)




