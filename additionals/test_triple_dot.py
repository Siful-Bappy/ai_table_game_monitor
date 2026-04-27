import cv2
import numpy as np

# -----------------------------------
# Step 1: Load two images
# -----------------------------------
template_path = "current.jpg"   # image at time t
current_path  = "current.jpg"    # image at time t+1

template_bgr = cv2.imread(template_path)
current_bgr  = cv2.imread(current_path)

if template_bgr is None or current_bgr is None:
    raise FileNotFoundError("Check image paths!")

# Convert to grayscale (optical flow works on single channel)
template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
current_gray  = cv2.cvtColor(current_bgr,  cv2.COLOR_BGR2GRAY)

# -----------------------------------
# Step 2: Compute dense optical flow
# -----------------------------------
flow = cv2.calcOpticalFlowFarneback(
    template_gray,
    current_gray,
    None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# flow shape = (H, W, 2)
H, W = template_gray.shape

print("Flow shape:", flow.shape)  # (H, W, 2)

# -----------------------------------
# Step 3: Create empty fx, fy arrays
# -----------------------------------
fx = np.zeros((H, W), dtype=np.float32)
fy = np.zeros((H, W), dtype=np.float32)

# -----------------------------------
# Step 4: Fill fx, fy manually (IMPORTANT PART)
# -----------------------------------
for y in range(H):        # loop over rows
    for x in range(W):    # loop over columns

        # flow[y, x] = [dx, dy]
        fx[y, x] = flow[y, x, 0]   # horizontal motion
        fy[y, x] = flow[y, x, 1]   # vertical motion

# -----------------------------------
# Step 5: Verify correctness
# -----------------------------------
# These MUST be True
print("fx equals flow[...,0]:", np.allclose(fx, flow[..., 0]))
print("fy equals flow[...,1]:", np.allclose(fy, flow[..., 1]))

# -----------------------------------
# Step 6: Inspect motion of a pixel
# -----------------------------------
test_x, test_y = 100, 100

print(f"\nPixel ({test_x},{test_y}) motion:")
print("dx =", fx[test_y, test_x])
print("dy =", fy[test_y, test_x])

# -----------------------------------
# Step 7: (Optional) Visualize fx, fy magnitude
# -----------------------------------
magnitude = np.sqrt(fx**2 + fy**2)
magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("Motion Magnitude", magnitude_norm.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
