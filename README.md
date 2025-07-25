# Subcortical Shape-based Prediction of Huntington’s Disease Progression (PIN score)

This repository contains the implementation of a deep learning framework for modeling Huntington's Disease (HD) progression using point cloud representations of subcortical brain structures. We developed and trained a PointNet-based architecture to learn anatomical shape descriptors that capture subtle morphometric deformations associated with HD severity.

## Overview

The core is a **discriminative PointNet model** trained to predict the Prognostic Index Normalized (PIN) score—a validated continuous measure of HD progression—from point cloud representations of segmented subcortical structures. 
These learned shape descriptors were then integrated into a conditional generative model (cVAE) for forecasting clinical and volumetric biomarkers at follow-up.

### Fork Acknowledgment

This PointNet implementation was **forked and adapted from the following repository**:

> https://github.com/itberrios/3D/tree/main/point_net

We gratefully acknowledge the original authors for their open-source codebase, which served as the foundation for our customizations.


