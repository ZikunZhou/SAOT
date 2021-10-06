import torch
import numpy as np
import os
import cv2

class Visualizer(object):
    def __init__(self):
        super(Visualizer, self).__init__()

    def visualize_single_map(self, map, output_size, path=None, name='default', show=False):
        """Visualize one single feature channel
        args:
            map - feature or response map that needs to be visualized, torch.Tensor, shape of map: [1, height, width]
            output_size - output size of the visualization, int or list([height, width])
            path - the path to save the visualization
            show - whether to show the visualization
        """
        # normalization
        max_value = torch.max(map)
        min_value = torch.min(map)
        if max_value == min_value:
            normed_map = map/max_value*255
        else:
            normed_map = (map-torch.min(map))/(torch.max(map)-torch.min(map))*255

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        heatmap = self.convert2heat(normed_map, output_size)
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), heatmap)

        if show:
            cv2.imshow(heatmap)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def visualize_multi_map_sum(self, map, output_size, path=None, name='defualt', show=False):
        """Visualize a multi-channle feature map by sum all the channels
        args:
            map - torch.Tensor, shape:[n, heith, width]
        """
        summed_map = torch.sum(map, dim=0, keepdim=True)
        self.visualize_single_map(summed_map, output_size, path, name, show)


    def convert2heat(self, normed_map, output_size, colormap=cv2.COLORMAP_JET):
        """
        convert the normalized map (torch.Tensor) to heatmap (numpy.array)
        args:
            normed_map - normalized map, torch.Tensor, shape: [1, height, width]
            output_size - output size of the visualization, list([height, width])
        """
        print(normed_map.shape)
        heatmap = normed_map.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)
        heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    def visualize_normed_image(self, normed_image, output_size, path=None, name='default', show=False):
        """Visualize the image that is normalized by the pytorch defualt settings:
                normalize_mean = [0.485, 0.456, 0.406]
                normalize_std = [0.229, 0.224, 0.225]
        args:
            normed_image - image that needs to be visualized, torch.Tensor, shape of map: [3, height, width]
            output_size - output size of the visualization, int or list([height, width])
            path - the path to save the visualization
            show - whether to show the visualization
        """

        image = self.unnorm(normed_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), image)

        if show:
            cv2.imshow(image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def visualize_tensor_image(self, image, output_size, path=None, name='default', show=False):
        """
        args:
            normed_image - image that needs to be visualized, torch.Tensor, shape of map: [3, height, width]
            output_size - output size of the visualization, int or list([height, width])
            path - the path to save the visualization
            show - whether to show the visualization
        """
        image = image.cpu().numpy().transpose((1,2,0)).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        image = image[63:223, 63:223,:]
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), image)

        if show:
            cv2.imshow(image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def visualize_tensor_image_with_box(self, image, box, output_size, path=None, name='default', show=False):
        """box - [x1, y1, w, h]
        """
        image = image.cpu().numpy().transpose((1,2,0)).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0,0,255), 3)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        image = image[63:223, 63:223,:]
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), image)

        if show:
            cv2.imshow(image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()


    def unnorm(self, normed_image):
        """
        unnorm the image tensor that is normalized according to the pytorch official procedure.
        """
        #print(normed_image.shape)
        image = (normed_image.cpu().numpy().transpose((1,2,0))+np.array([0.229,0.224,0.225]))*np.array([0.485,0.456,0.406])
        image = (image * 255+20).clip(0,255).astype(np.uint8)
        return image

    def visualize_normed_image_with_box(self, normed_image, box, output_size, path=None, name='default', show=False):
        """
        box - [x1, y1, w, h]
        """
        image = self.unnorm(normed_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0,0,255), 3)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), image)

        if show:
            cv2.imshow(image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def visualize_normed_image_with_box_dot(self, normed_image, box, cls, points, output_size, path=None, name='default', show=False):
        """
        box - [x1, y1, w, h]
        """
        image = self.unnorm(normed_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0,0,255), 3)

        for i, point in enumerate(points.to(torch.int)):
            if cls[i]>0:
                cv2.circle(image, tuple(point.numpy().tolist()), 1, [0,0,255], 3)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)

        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), image)

        if show:
            cv2.imshow(image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def merge_normed_image_and_single_map(self, normed_image, map, output_size, path=None, name='default', show=False):
        """
        args:
            normed_image - image that needs to be visualized, torch.Tensor, shape of map: [3, height, width]
            map - feature or response map that needs to be visualized, torch.Tensor, shape of map: [1, height, width]
            output_size - output size of the visualization, int or list([height, width])
            path - the path to save the visualization
            show - whether to show the visualization
        """
        # normalization
        max_value = torch.max(map)
        min_value = torch.min(map)
        if max_value == min_value:
            normed_map = map/max_value*255
        else:
            normed_map = (map-torch.min(map))/(torch.max(map)-torch.min(map))*255

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        heatmap = self.convert2heat(normed_map, output_size)

        image = self.unnorm(normed_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        blend = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), blend)

        if show:
            cv2.imshow(blend)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def merge_tensor_image_and_single_map(self, image, map, output_size, path=None, name='default', show=False):
        """
        args:
            image - image that needs to be visualized, torch.Tensor, shape of map: [3, height, width]
            map - feature or response map that needs to be visualized, torch.Tensor, shape of map: [1, height, width]
            output_size - output size of the visualization, int or list([height, width])
            path - the path to save the visualization
            show - whether to show the visualization
        """
        # normalization
        max_value = torch.max(map)
        min_value = torch.min(map)
        if max_value == min_value:
            normed_map = map/max_value*255
        else:
            normed_map = (map-torch.min(map))/(torch.max(map)-torch.min(map))*255

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        heatmap = self.convert2heat(normed_map, output_size)

        image = image.cpu().numpy().transpose((1,2,0)).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
        blend = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, '{:s}.jpg'.format(name)), blend)

        if show:
            cv2.imshow(blend)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
