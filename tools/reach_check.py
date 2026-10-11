"""
Reach check for the Yuktii car painting cell.

Frame: origin at the center of the car footprint on the floor.
  x = along the car length, y = across the car width, z = up. Units: meters.

Edit the parameters below, run, then rotate the 3D view with the mouse.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters (edit these) ----------------
CAR_L, CAR_W, CAR_H = 4.8, 1.9, 1.6     # car body box
BOOTH_L, BOOTH_W = 8.0, 5.0             # central booth from the reference layout
SIDE_GAP = 1.                         # car side to robot base
BASE_X = 0.0                            # 0 = level with the middle of the car
SHOULDER_HEIGHT = 1.0                   # height of the arm's shoulder joint above the floor
REACH = 2.7                             # arm reach, measured from the shoulder
SAVE_FIGURE = True                      # also writes reach_check.png for the report
# ----------------------------------------------------------

base = np.array([BASE_X, -(CAR_W / 2 + SIDE_GAP), 0.0])
shoulder = base + np.array([0.0, 0.0, SHOULDER_HEIGHT])

# Eight corners of the car box
xs = [-CAR_L / 2, CAR_L / 2]
ys = [-CAR_W / 2, CAR_W / 2]
zs = [0.0, CAR_H]
corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Car box edges: connect corners that differ in exactly one coordinate
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        if np.sum(corners[i] != corners[j]) == 1:
            ax.plot(*zip(corners[i], corners[j]), color="tab:blue", lw=2)

# Label corners so you can refer to them
for k, c in enumerate(corners):
    ax.scatter(*c, color="tab:blue", s=25)
    ax.text(*c, f"  C{k}", fontsize=8)

# Booth floor outline
bx, by = BOOTH_L / 2, BOOTH_W / 2
outline = np.array([[-bx, -by, 0], [bx, -by, 0], [bx, by, 0], [-bx, by, 0], [-bx, -by, 0]])
ax.plot(outline[:, 0], outline[:, 1], outline[:, 2], color="gray", ls="--", label="booth walls")

# Robot base and shoulder
ax.scatter(*base, color="tab:red", s=80, marker="s", label="robot base")
if SHOULDER_HEIGHT > 0:
    ax.plot(*zip(base, shoulder), color="tab:red", lw=3)
    ax.scatter(*shoulder, color="tab:red", s=40, label="shoulder")

# Reach sphere (wireframe) centered on the shoulder
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
sx = shoulder[0] + REACH * np.cos(u) * np.sin(v)
sy = shoulder[1] + REACH * np.sin(u) * np.sin(v)
sz = shoulder[2] + REACH * np.cos(v)
sz = np.where(sz < 0, np.nan, sz)       # hide the part of the sphere below the floor
ax.plot_wireframe(sx, sy, sz, color="tab:orange", alpha=0.25, lw=0.6)

ax.set_xlabel("x, along car (m)")
ax.set_ylabel("y, across car (m)")
ax.set_zlabel("z, up (m)")
ax.set_xlim(-bx, bx)
ax.set_ylim(-by - 0.5, by)
ax.set_zlim(0, max(CAR_H, shoulder[2] + REACH))
ax.set_box_aspect((BOOTH_L, BOOTH_W + 0.5, max(CAR_H, shoulder[2] + REACH)))
ax.set_title(f"Reach {REACH} m from shoulder at {np.round(shoulder, 2)}")
ax.legend(loc="upper left")

# Print the coordinates so you can do the distance math yourself
print("Shoulder position:", np.round(shoulder, 3))
for k, c in enumerate(corners):
    print(f"C{k}: {np.round(c, 3)}")

if SAVE_FIGURE:
    plt.savefig("reach_check.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDistance from shoulder to each corner:")
for k, c in enumerate(corners):
    d = np.linalg.norm(c - shoulder)
    status = "inside" if d <= REACH else "outside"
    print(f"C{k}: {d:.2f} m  ({status}, margin {REACH - d:+.2f} m)")