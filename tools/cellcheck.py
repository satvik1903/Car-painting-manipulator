"""
Yuktii car painting cell: 3D layout planner.

Draws the reference booth, the car, and one robot mounting option, then checks
which points on the car's paintable surfaces the robot can actually paint.

Frame: origin at the center of the car footprint on the floor.
  x = along the car (conveyor direction), y = across the car, z = up. Meters.

Run:  python3 cell_plan.py fixed | side_rail | u_rail | gantry
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# ======================= Parameters (edit these) =======================
LAYOUT = sys.argv[1] if len(sys.argv) > 1 else "side_rail"

CAR_L, CAR_W, CAR_H = 4.8, 1.9, 1.6
GROUND_CLEARANCE = 0.2     # underbody is not painted, so side panels start here

# Arm and tool
REACH = 2.7                # shoulder to wrist flange, max
MIN_REACH = 0.6            # closer than this and the arm folds on itself
TOOL_LENGTH = 0.3          # flange to spray gun nozzle
STANDOFF = 0.25            # nozzle to car surface while painting

# Floor mounted layouts (fixed, side_rail, u_rail)
SIDE_GAP = 1.0             # car side to robot base
END_GAP = 1.0              # car bumper to robot base (u_rail only)
SHOULDER_HEIGHT = 1.0      # shoulder above floor for a floor mounted arm

# Overhead layout (gantry): arm hangs upside down from a beam over the car
GANTRY_Y = 0.0             # beam position across the car (0 = centerline)
GANTRY_SHOULDER_Z = 3.0    # shoulder height of the inverted arm

RAIL_STEP = 0.1            # spacing of sampled base positions along a rail
SURFACE_STEP = 0.1         # spacing of sampled points on the car surface
# ========================================================================

L2, W2 = CAR_L / 2, CAR_W / 2


def base_path(layout):
    """Shoulder positions the robot can occupy, shape (M, 3)."""
    y_near = -(W2 + SIDE_GAP)
    if layout == "fixed":
        return np.array([[0.0, y_near, SHOULDER_HEIGHT]])
    if layout == "side_rail":
        xs = np.arange(-L2, L2 + 1e-9, RAIL_STEP)
        return np.column_stack([xs, np.full_like(xs, y_near), np.full_like(xs, SHOULDER_HEIGHT)])
    if layout == "u_rail":
        x_end = L2 + END_GAP
        y_far = W2 + SIDE_GAP
        xs = np.arange(-L2, x_end + 1e-9, RAIL_STEP)
        ys = np.arange(y_near, y_far + 1e-9, RAIL_STEP)
        near = np.column_stack([xs, np.full_like(xs, y_near)])
        end = np.column_stack([np.full_like(ys, x_end), ys])
        far = np.column_stack([xs[::-1], np.full_like(xs, y_far)])
        xy = np.vstack([near, end, far])
        return np.column_stack([xy, np.full(len(xy), SHOULDER_HEIGHT)])
    if layout == "gantry":
        xs = np.arange(-L2, L2 + 1e-9, RAIL_STEP)
        return np.column_stack([xs, np.full_like(xs, GANTRY_Y), np.full_like(xs, GANTRY_SHOULDER_Z)])
    raise ValueError(f"Unknown layout '{layout}'")


def car_surface_points():
    """Sample the five paintable faces. Returns points, outward normals, face names."""
    s = SURFACE_STEP
    xs = np.arange(-L2, L2 + 1e-9, s)
    ys = np.arange(-W2, W2 + 1e-9, s)
    zs = np.arange(GROUND_CLEARANCE, CAR_H + 1e-9, s)
    faces = []

    X, Y = np.meshgrid(xs, ys)
    faces.append(("roof", np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, CAR_H)]), [0, 0, 1]))
    X, Z = np.meshgrid(xs, zs)
    faces.append(("near side", np.column_stack([X.ravel(), np.full(X.size, -W2), Z.ravel()]), [0, -1, 0]))
    faces.append(("far side", np.column_stack([X.ravel(), np.full(X.size, W2), Z.ravel()]), [0, 1, 0]))
    Y, Z = np.meshgrid(ys, zs)
    faces.append(("front", np.column_stack([np.full(Y.size, L2), Y.ravel(), Z.ravel()]), [1, 0, 0]))
    faces.append(("rear", np.column_stack([np.full(Y.size, -L2), Y.ravel(), Z.ravel()]), [-1, 0, 0]))

    pts = np.vstack([f[1] for f in faces])
    nrm = np.vstack([np.tile(f[2], (len(f[1]), 1)) for f in faces])
    names = np.concatenate([[f[0]] * len(f[1]) for f in faces])
    return pts, nrm.astype(float), names


def reachable(points, normals, shoulders):
    """A point is paintable if the wrist flange, placed so the gun points straight
    at the surface from the standoff distance, is within the arm's reach band
    from at least one shoulder position."""
    flange = points + normals * (STANDOFF + TOOL_LENGTH)
    d = np.linalg.norm(flange[:, None, :] - shoulders[None, :, :], axis=2)
    ok = (d <= REACH) & (d >= MIN_REACH)
    return ok.any(axis=1)


def box_edges(ax, x0, x1, y0, y1, z0, z1, **kw):
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(c[i] != c[j]) == 1:
                ax.plot(*zip(c[i], c[j]), **kw)


def floor_rect(ax, x0, x1, y0, y1, **kw):
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [0] * 5, **kw)


# ------------------------------- Build -------------------------------
shoulders = base_path(LAYOUT)
pts, nrm, names = car_surface_points()
ok = reachable(pts, nrm, shoulders)

fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(111, projection="3d")

# Reference layout footprint (approximate, from the PDF drawing)
floor_rect(ax, -4, 4, -2.5, 2.5, color="black", lw=1.5)                      # paint booth 8 x 5
floor_rect(ax, -6, -4, -2.5, 2.5, color="gray", ls="--")                     # entry zone 2 x 5
floor_rect(ax, 4, 6, -2.5, 2.5, color="gray", ls="--")                       # exit zone 2 x 5
for x0, x1 in [(-5.75, -2.75), (-2.0, 0.5), (1.75, 5.75)]:                   # support rooms
    floor_rect(ax, x0, x1, 2.5, 5.0, color="silver", ls=":")
ax.text(-5, 0, 0, "ENTRY", color="gray", fontsize=8, ha="center")
ax.text(5, 0, 0, "EXIT", color="gray", fontsize=8, ha="center")
ax.quiver(-6.8, 0, 0.05, 0.7, 0, 0, color="gray", arrow_length_ratio=0.3)
ax.quiver(6.1, 0, 0.05, 0.7, 0, 0, color="gray", arrow_length_ratio=0.3)

# Car and surface coverage
box_edges(ax, -L2, L2, -W2, W2, 0, CAR_H, color="tab:blue", lw=1)
ax.scatter(*pts[ok].T, color="tab:green", s=3, label="paintable")
ax.scatter(*pts[~ok].T, color="tab:red", s=3, label="not paintable")

# Robot path
ax.plot(*shoulders.T, color="tab:orange", lw=3, label="shoulder path")
if LAYOUT == "gantry":
    top = GANTRY_SHOULDER_Z + 0.3
    for x in (-L2 - 0.6, L2 + 0.6):                     # portal legs beyond the bumpers
        ax.plot([x, x], [GANTRY_Y, GANTRY_Y], [0, top], color="dimgray", lw=2)
    ax.plot([-L2 - 0.6, L2 + 0.6], [GANTRY_Y] * 2, [top] * 2, color="dimgray", lw=4, label="gantry beam")
else:
    ax.plot(shoulders[:, 0], shoulders[:, 1], np.zeros(len(shoulders)), color="dimgray", lw=4, label="rail / base")

ax.set_xlabel("x, along car (m)")
ax.set_ylabel("y, across car (m)")
ax.set_zlabel("z, up (m)")
ax.set_xlim(-7, 7)
ax.set_ylim(-3, 5.5)
ax.set_zlim(0, 4)
ax.set_box_aspect((14, 8.5, 4))
ax.set_title(f"Layout: {LAYOUT}   |   reach {REACH} m, standoff {STANDOFF} m, tool {TOOL_LENGTH} m")
ax.legend(loc="upper left", markerscale=4)

# ----------------------------- Report -----------------------------
print(f"\nLayout: {LAYOUT}")
for face in ["roof", "near side", "far side", "front", "rear"]:
    m = names == face
    print(f"  {face:<10} {100 * ok[m].mean():5.1f}% paintable")
print(f"  {'TOTAL':<10} {100 * ok.mean():5.1f}%")

plt.savefig(f"cell_plan_{LAYOUT}.png", dpi=150, bbox_inches="tight")
plt.show()
