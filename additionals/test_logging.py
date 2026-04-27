# import logging
# import time

# # 1. Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s-%(levelname)s-%(message)s'
# )

# # 2. Create a logger
# logger = logging.getLogger("PRACTICE")

# # 3. Simulate a running program (like video frames)
# for frame in range(1, 6):
#     logger.info(f"Processing frame {frame}")
    
#     # simulate work
#     time.sleep(0.5)
    
#     if frame == 3:
#         logger.warning("Something looks unusual at frame 3")
    
#     if frame == 5:
#         logger.error("Fake error at last frame")

# logger.info("Program finished")


import numpy as np
def decompose_affine(M):

    """Affine matrix to  angle, scale, translation
    
       Y = s*R X + t
       
       assuming center at (0,0)
       
       return angle in degree, scale, translation 
    
    """
    tx, ty = M[0,2], M[1,2]
    print(f"decompose_affine: tx={tx}, ty={ty}")
    a, b = M[0,0], M[0,1]
    print(f"decompose_affine: a={a}, b={b}")
    scale = np.sqrt(a*a + b*b)
    print(f"decompose_affine: scale={scale}")
    angle = np.degrees(np.arctan2(b, a))
    print(f"decompose_affine: angle={angle}")
    return angle, scale, (tx, ty)

M = np.array([
    [1, 0, 100],
    [0, 1, 50]
], dtype=float)
decompose_affine(M)