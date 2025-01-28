import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from dm_control import mujoco as dm_mujoco
from mujoco import FatalError as mujocoFatalError

from mjcf2o3d.file_utils import get_temp_filepath
from mjcf2o3d.scanner.env_utils import VisionSensorOutput


#SCENE_BOUNDS=((-1.4, -0.2, -0.1), (1.7, 1.2, 1.1))
class MujocoSimEnv:
    """
    Base environment for all tasks. Loads from a mujoco xml file and accesses the simulation
    via dm_control Physics engine. Notice how some methods are not implemented, these are
    specific to each task. See task_[task_name].py for more details.
    """

    def __init__(
            self,
            filepath: str,
            render_cameras: List[str] = [f"face_{i}" for i in range(16)],
            bounds=None,
            image_hw: Tuple = (480, 480),
    ):
        self.bounds = bounds

        # print(filepath)
        self.xml_file_path = filepath
        self.physics = dm_mujoco.Physics.from_xml_path(filepath)

        # check rendering options
        self.render_buffers = dict()
        for cam in render_cameras:
            try:
                self.physics.render(camera_id=cam, height=image_hw[0], width=image_hw[1])
            except Exception as e:
                print("Got Error: ", e)
                print("Camera {} does not exist in the xml file".format(cam))
            self.render_buffers[cam] = deque(maxlen=3000)
        self.render_cameras = render_cameras
        self.image_hw = image_hw

        self.reset()

    def reset(self):
        self.physics.reset()
        self.clear_camera_buffer()
        self.render_all_cameras()

    def clear_camera_buffer(self):
        self.render_buffers = {cam: deque(maxlen=1000) for cam in self.render_cameras}

    def export_render_to_video(self, output_name="task_video", out_type="gif", fps=20, concat=True, video_duration=0):
        render_steps = len(self.render_buffers[self.render_cameras[0]])
        assert render_steps > 0 and all([len(self.render_buffers[cam]) == render_steps for cam in self.render_cameras]), \
            "Render buffers are not all the same length, got lengths: {}".format(
                [len(self.render_buffers[cam]) for cam in self.render_cameras])
        assert out_type in ["gif", "mp4"], "out_type must be either gif or mp4"
        all_imgs = []
        for t in range(render_steps):
            images = [self.render_buffers[cam][t] for cam in self.render_cameras]
            if concat:
                images = np.concatenate(images, axis=1)
            else:
                images = images[0]
            all_imgs.append(images)
        if out_type == "gif":
            all_imgs = [Image.fromarray(img) for img in all_imgs]
            output_name += ".gif" if ".gif" not in output_name else ""
            if video_duration > 0:
                # ignore fps, use video duration instead
                duration = int(video_duration / render_steps * 1000)
            else:
                duration = int(1000 / fps)
            all_imgs[0].save(
                output_name,
                save_all=True,
                append_images=all_imgs[1:],
                duration=duration,
                loop=0
            )
        elif out_type == "mp4":
            output_name += ".mp4" if ".mp4" not in output_name else ""
            w, h = all_imgs[0].shape[:2]
            if video_duration > 0:
                # ignore fps, use video duration instead
                fps = int(render_steps / video_duration)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(
                output_name, fourcc, fps, (h, w))
            for img in all_imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
            video.release()
        print('Video gif, total {} frames, saved to {}'.format(render_steps, output_name))

    def render_camera(self, camera_id, height=480, width=600):
        img_arr = self.physics.render(
            camera_id=camera_id, height=height, width=width,
        )
        self.render_buffers[camera_id].append(img_arr)
        return img_arr

    def render_all_cameras(self, save_img=False, output_name="render.jpg", show=False):
        imgs = []
        for cam_id in self.render_cameras:
            img_arr = self.render_camera(cam_id, height=self.image_hw[0], width=self.image_hw[1])
            imgs.append(img_arr)
        imgs = np.concatenate(imgs, axis=1)
        if show:
            plt.imshow(imgs)
            plt.show()
        if save_img:
            tosave = Image.fromarray(imgs)
            tosave.save(output_name)
        return imgs

    def render(
            self, max_retries: int = 100) -> Dict[str, VisionSensorOutput]:
        outputs = {}
        for cam_name in self.render_cameras:
            cam = self.physics.model.camera(cam_name)
            cam_data = self.physics.data.camera(cam_name)
            cam_pos = cam_data.xpos.reshape(3)
            cam_rotmat = cam_data.xmat.reshape(3, 3)
            for i in range(max_retries):
                try:
                    # NOTE: rgb render much more expensive than others
                    # If optimizing, look into disable rgb rendering for
                    # passes which are not needed
                    rgb = self.physics.render(
                        height=self.image_hw[0],
                        width=self.image_hw[1],
                        depth=False,
                        camera_id=cam.id,
                    )
                    depth = self.physics.render(
                        height=self.image_hw[0],
                        width=self.image_hw[1],
                        depth=True,
                        camera_id=cam.id,
                    )
                    segmentation = self.physics.render(
                        height=self.image_hw[0],
                        width=self.image_hw[1],
                        depth=False,
                        segmentation=True,
                        camera_id=cam_name,
                    )

                    outputs[cam_name] = VisionSensorOutput(
                        rgb=rgb,
                        depth=depth,
                        pos=(cam_pos[0], cam_pos[1], cam_pos[2]),
                        rot_mat=cam_rotmat,
                        fov=float(cam.fovy[0]),
                    )
                    break

                except mujocoFatalError as e:
                    if i == max_retries - 1:
                        raise e
                    time.sleep(5)
        return outputs

    def get_point_cloud(self):
        sensor_outputs = self.render()
        point_clouds = [
            sensor_output.point_cloud if not self.bounds else sensor_output.point_cloud.filter_bounds(bounds=self.bounds)
            for sensor_output in sensor_outputs.values()
        ]
        point_cloud = sum(point_clouds[1:], start=point_clouds[0])
        return point_cloud

def scan(file, root_position, num_cameras, max_distance):
    bounds = (root_position-max_distance, root_position+max_distance)

    env = MujocoSimEnv(file, render_cameras=[f"camera_{i}" for i in range(num_cameras)], bounds=bounds)

    gif_filepath = get_temp_filepath(ext="")
    env.export_render_to_video(gif_filepath)
    #img = Image.open(gif_filepath+".gif")

    #numpydata = np.asarray(img)

    pcd = env.get_point_cloud().to_open3d()

    return pcd, gif_filepath+".gif"

    import open3d as o3d

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pcd)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    env = MujocoSimEnv("./mjcf2o3d/robot_with_cameras.mjcf")

    env.export_render_to_video("./yay")
    #exit()


    pcd = env.get_point_cloud().to_open3d()

    import open3d as o3d

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(pcd)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.poll_events()
    visualizer.update_renderer()

    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    visualizer.run()