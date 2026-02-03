---
title: "Updates & Notes"
author: "Cameron"
date: \today
bibliography: citations.bib
---

# Why Inverse Dynamics → Forward Kinematics Pipeline

Learning ID in latent space is (probably) easier than learning FK

* Invertibility $\rightarrow$ bijective
* bijective $\rightarrow$ *no* information loss
	* inputs can be reconstructed from outputs

* Inspired by [@brandfonbrener2023inverse]

# Bounds on FK Errors

* Function of ID errors and Lipschitz of FK?

# Total Pipeline:

* Learn ID (good)
* Get embeddings from ID (good)
* Train FK on embeddings (good)
* Train decoder on embeddings (?)
	* Currently, decoder has kind of high loss, but only with an early-stopped ID embedding

# Upcoming

* Better exploration policy
	* Random? Brownian noise in action space?
* Planning

# WCP
* ~ halfway there
