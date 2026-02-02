---
title: "Updates & Notes"
author: "Cameron"
date: \today
---

# Inverse Dynamics → Forward Kinematics Pipeline

**Idea**: Learn control representations instead

1. Train ID model: $(I_t, I_{t+1}) \to (v_L, v_R)$ [supervised]
2. Apply FK: $(v_L, v_R) \to (\dot{x}_{body}, \dot{\theta}_{body})$ [deterministic]
3. Integrate for pose predictions

Does this preserve enough information?

* Invertibility $\rightarrow$ bijective
* bijective $\rightarrow$ *no* information loss
	* inputs can be reconstructed from outputs

# Bijectivity Analysis (differential drive example)

(in the body frame)

$$\begin{aligned}
\dot{x}_{body} &= v = \frac{v_L + v_R}{2} \\
\dot{y}_{body} &= 0 \\
\dot{\theta} &= \omega = \frac{v_R - v_L}{L}
\end{aligned}$$

# Bijectivity Analysis: Bicycle Model

**Bicycle model dynamics** (body frame):

$$\begin{aligned}
\dot{x}_{body} &= v \\
\dot{y}_{body} &= 0\\
\dot{\theta} &= \frac{v \tan(\delta)}{L}
\end{aligned}$$

where $v$ is velocity, $\delta$ is steering angle, $L$ is wheelbase.

**Controls**: $(v, \delta)$ or $(\dot{v}, \delta)$ depending on whether velocity is controlled directly

# **Inverse dynamics** (body frame):

$$\begin{aligned}
v &= \dot{x}_{body} \\
\delta &= \arctan\left(\frac{L \dot{\theta}}{\dot{x}_{body}}\right)
\end{aligned}$$

**Bijectivity**: $(\dot{x}_{body}, \dot{\theta}) \leftrightarrow (v, \delta)$
