# Search and Sample Return

![rover][image3]

[//]: # (Image References)

[image1]: calibration_images/example_grid1.jpg
[image2]: calibration_images/example_rock1.jpg
[image3]: misc/rover_image.jpg
[image4]: misc/grid_perspective_transform.png
[image5]: misc/autonomous_mode.jpg
[image6]: misc/grid_threshold.png
[image7]: misc/grid_with_direction.jpg
[image8]: misc/rock_threshold.png
[image9]: misc/vision.jpg
[image10]: misc/simulator_setting.png
[image11]: misc/simulator.png




## Detailed Project Analysis

### Notebook Analysis

#### 1. Image Analysis

The `perspect_transform` function turns a Rover perspective image to a top-down view.

```
def perspect_transform(img, src, dst):  
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))

    return warped
```

For example,

![Before perspective transform][image1]

![After perspective transform][image4]

The `color_thresh` function will return a binary image where it is 1 when that pixel color value in the `img` is above the given RGB threshold.

```
def color_thresh(img, rgb_thresh=(160, 160, 160)):

	color_select = np.zeros_like(img[:,:,0])

	above_thresh = (img[:,:,0] > rgb_thresh[0]) \
				& (img[:,:,1] > rgb_thresh[1]) \
				& (img[:,:,2] > rgb_thresh[2])
	color_select[above_thresh] = 1

	return color_select
```

The default RBG value `R=G=B=160` is found to perform well in terms of determing navigable terrain.

For example,

![Before color threshold][image4]

![After color threshold][image6]


```
def obstacle_thresh(img):
    return color_thresh(img, above=False)
```

I've added the find_rocks function. To identify the sample rocks, the RGB threshold of `R>110, G=110, B<50` is found to be effective.

```
def find_rocks(img, levels=(110, 110, 50)):
    rockpix = ((img[:,:,0] > levels[0]) \
               & (img[:,:,1] > levels[1]) \
               & (img[:,:,2] < levels[2]))

    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1

    return color_select
```

For example,

![Rock Image][image2]

![Rock Threshold][image8]

#### 2. Mapping
The `process_image(img)` function takes advantage of all the image analysis functions described above and performs the mapping. It maps identified pixels navigable terrain, obstacles and rock samples into a worldmap.

To achieve this, there are 7 main steps.

First, I define the source and destination points for perspective transform. These parameters are found to be effective:

```
dst_size = 5
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset], \
                          [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset], \
                          [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], \
                          [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset]])
```

Then, I apply perspective transform on the vision image received from Rover. This will gives us a top-down view of the environment, which is useful when constructing the map.

```
warped, mask = perspect_transform(img, source, destination)
```

After that, I will apply the color threshold to identify navigable terrain samples.

```
threshed = color_thresh(warped)
```
The obstacle map is just the threshed map minus one times the mask. So basically I'm going to get back ones everywhere the threshed map was zero and them multiply that by my mask which has zeros where that's outside the camera field of view.

```
obs_map = np.absolute(np.float32(threshed) - 1) * mask
```

Finally I call the find_rocks function to identify rocks.

```
rock_map = find_rocks(warped, levels=(110, 110, 50))
```

To update the worldmap, we need to first transform our post-analysis image to be rover-centric so the position of rover is at position `(0,0)` instead of the bottom middle of the image. Then, we need to scale the image smaller the fit in our worldmap.

These are accomplished by mainly two supporting functions: `rover_coords` which converts from image coordinates to rover coordinates, and `pix_to_world` which applies rotation, translation, and clipping. You can view more details on these functions by visiting the `perception.py` script.

Here is an example of the entire process:

![Full Mapping Process][image7]

### Autonomous Navigation and Mapping

#### 1. Perception Step

The perception of the Rover is handled in the `perception_step()` (at the bottom of the `perception.py` script). In this step, I will perform relevant image analysis and mapping, update the Rover vision and worldmap, and prepare data to be used for the decision step next.

This step is highly similar to the `process_image` described above in the "Notebook Analysis - Mapping" section. There are only a few modifications.

First, I only update the worldmap when Rover has a normal view. In other words, not when it is stuck in a rock, beyond the simulated world boundry, or got flipped and looking at the sky. I accomplish this by updating the world map only when Rover roll and pitch are both smaller than `0.7` or bigger than `359.5`.

```
roll, pitch = Rover.roll, Rover.pitch
if roll <= 0.7 or roll >= 359.5:
        if pitch <= 0.7 or pitch >= 359.5:
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 255
        Rover.worldmap[rock_y_world, rock_x_world, 1] += 255
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 255
```

Second, I update a vision map so users can see what Rover is perceiving, where red indicates obstacles; blue indicates navigable terrain, and green/yellow indicates the rock.

```
Rover.vision_image[:,:,0] = threshed * 255
Rover.vision_image[:,:,1] = obs_map * 255
Rover.vision_image[:,:,2] = rock_map * 255
```

For example,

![Vision Map][image9]

Third, I also calculate the angles and distances of both navigable terrain and rock in polar coordinates via `to_polar_coords` (also defined in `perception.py` script). These data will help make the decision of where to go in the decision step.

```
dist, Rover.nav_angles = to_polar_coords(xpix, ypix)
 _ , Rover.rock_angles = to_polar_coords(rock_x_rover, rock_y_rover)

```

#### 2. Decision Step

The decision step of the Rover is handled in the `decision_step()` (in `decision.py`). In this step, I will write conditional logic to control the behaviors of the Rover.

There is a moderate amount of code written for this step, so I will only explain the high level strategy here.

**In the 'forward' mode**

When the navigable angles are big and wide enough -- the boundry is determined by a parameter called `random_direction_angles`, I set the steer to a random angle sampled from a normal distribution (centered at the navigable angles mean, plus or minus 3 based on the direction it's going right now, with standard deviation of 3) with the probability of 40%. The rationale behind this complicated random sampling is to add randomness so Rover has the chance of going to different places when come to a big open space, instead of always going to the same direction and never be able to explore other places.

Here is an example of the code used for idea 3:

```
if len(Rover.nav_angles) >= Rover.random_direction_angles:
    if np.random.random() <= 0.4:
        if Rover.steer >= 0:
            Rover.steer = np.clip(np.random.normal(Rover.steer - 3, 3), -15, 15)
        else:
            Rover.steer = np.clip(np.random.normal(Rover.steer + 3, 3), -15, 15)
```


**In the 'stop' mode**

1. If Rover is still moving, let it stop.
2. If Rover is stopped and doesn't have much navigable terrain ahead of it, let it turn around in place. This is accomplished by setting throttle to 0, brake to 0, but steer to -15.
3. If Rover is stopped and now it has enough navigable terrain ahead after turning, we let it move again by setting it to 'forward' mode.


**Getting Unstuck**

Someimes a Rover can get stuck for various reasons. Whenever the Rover is at a speed slower than 0.2 larger than -0.2 (aka. going backwards) but in 'forward' mode, I start counting how long it has been slower than 0.2 speed. Every two seconds, I let Rover turn the steer more and more, switch between postive direction and negataive direction, and accelerating backwards. I stop the throttle and turn again before accelerating if Rover is trapped for more than 5 seconds.

#### 3. Autonomous Mode, Results, and Potential Improvements

The simulator I used for development is: **640x480 resolution, "Good" graphic quality, and 38 frames per second.** Under these settings, I only need to swtich to manual mode to get Rover unstuck in very rare occasions, while getting 66.4% of the world mapped at 78.4% fidelity. Rover is also able to find rocks.

![Autonomous Result][image5]

_**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!**_

I would like to revisit this project in the future, and try some end to end deep learning for this rover. I would also like to fix my current approach in the near future by making my rover a wall crawler. This project was a lot of fun!
