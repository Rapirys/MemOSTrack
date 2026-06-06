**Extending OSTrack with GRU-Based Memory Tokens for Visual Object Tracking**

Student's name and surname

Zagreb, June 2026

# Acknowledgements

# Table of Contents

Introduction

1\. Background and Related Work

2\. OSTrack Baseline

3\. Proposed Memory-Augmented OSTrack

4\. Sequence-Based and Inference-Like Training

5\. Auxiliary Memory Supervision

6\. Experimental Setup

7\. Results and Discussion

Conclusion

References

Abstract

Summary

Abbreviations

Appendix

# Introduction

Visual object tracking is the task of locating a target object through a video after the
target has been specified in the first frame. The input is simple: the tracker receives
an initial bounding box, and in each subsequent frame it predicts the new target
position. The difficulty is that the object can move, change scale, deform, become
partially occluded, or appear in a cluttered background. A tracker must therefore
combine an initial target description with current visual evidence and, ideally, with
temporal information collected during the sequence.

This thesis investigates a memory extension of OSTrack, a one-stream transformer
tracker. OSTrack is a strong baseline because it jointly processes template and search
tokens in a single transformer backbone. The goal of this work is not to replace the
whole tracker, but to test whether explicit recurrent memory tokens can be inserted into
the OSTrack token stream and trained on video sequences in a meaningful way.

The proposed experimental model is called MemOSTrack. It extends the original
template-search token sequence with memory tokens and updates these tokens using a
two-stage gated recurrent unit (GRU) mechanism. The first recurrent update combines the
candidate memory produced by the current transformer layer with memory from the previous
frame. The second recurrent update combines the result with memory from the previous
transformer layer. This design tries to give the tracker both temporal continuity and
layer-wise memory refinement.

The work is not limited to an architectural change. A recurrent tracker cannot be
trained correctly using only independent template-search frame pairs, because the
recurrent state would not receive a meaningful ordered sequence. For this reason, the
sampling pipeline was first rewritten to return video frames as ordered sequences. It
was later rewritten again to use dynamic search cropping, where the next search crop is
generated from the previous prediction rather than from the current ground-truth target
position. This final version is more similar to real inference, where the tracker does
not know the target box after initialization.

The thesis also studies a practical problem that appears when adding memory to a strong
baseline: the new memory branch can be ignored. Auxiliary memory loss heads were
explored to make memory tokens predictive, and template blurring was considered as a
regularization strategy that weakens over-reliance on the clean initial template. These
ideas do not guarantee improvement, but they make the thesis more than a simple
implementation exercise: the work examines how a memory pathway can be encouraged to
carry useful temporal information.

The final evaluation compares MemOSTrack-256 + CE with the reported OSTrack-256 + CE
and OSTrack-384 + CE baselines. The experimental model obtained AO 0.729, SR0.50 0.823,
and SR0.75 0.693. It is better than OSTrack-256 + CE by 0.019 AO, 0.019 SR0.50, and
0.011 SR0.75, but remains below OSTrack-384 + CE by 0.008 AO, 0.009 SR0.50, and 0.015
SR0.75 \[13\]. Therefore, the result is positive relative to the same-resolution 256
baseline, but not competitive with the stronger high-resolution 384 baseline. This
distinction is central to the interpretation of the thesis.

The main contributions of the thesis are: introduction of memory tokens into the OSTrack
token sequence; implementation of a two-stage GRU memory update; rewriting of the
training sampler for ordered video sequences; implementation of inference-like dynamic
search cropping; exploration of memory loss heads and template blurring; and
experimental comparison with CE-enabled OSTrack baselines at 256 and 384 search
resolutions.

# 1. Background and Related Work

## 1.1 Single-object tracking

Single-object tracking is a sequential computer vision problem. In the first frame of a
video, the target is identified by a bounding box. In later frames, the tracker predicts
the target location without receiving additional manual annotation. This distinguishes
tracking from ordinary object detection: the class of the object is not enough, because
the tracker must follow a particular instance. If several similar objects are present,
the correct output is the original target, not merely any object of the same category.

*Given frames I_1, I_2, ..., I_T and the initial box B_1, a tracker predicts boxes
\hat{B}\_t for t \> 1.*

The tracker may also maintain an internal state. A memory-free tracker can be written as
a function of the template and the current search region, while a recurrent tracker
additionally uses a state carried from previous frames. This distinction is central to
this thesis, because MemOSTrack explicitly introduces recurrent memory tokens into
the OSTrack backbone.

Tracking is difficult because the visual appearance of the target is not fixed. The
target may rotate, deform, change illumination, become blurred by motion, or be partly
hidden. The background can contain objects that look similar to the target. The camera
may move, producing large apparent target displacement between frames. A tracking method
must therefore balance stability and adaptability: it should not drift to a distractor,
but it must adapt when the true target changes appearance.

Single-object tracking (SOT) is the task of localizing the same target object throughout
a video sequence, given only its bounding box in the first frame. Early tracking methods
were mainly based on hand-crafted appearance and motion models, including mean-shift
tracking \[1\], optical flow-based methods \[2\], particle filters \[3\], and template
matching approaches \[4\]. These methods were often efficient and conceptually simple,
but they struggled under large appearance changes, occlusion, background clutter, and
scale variation. Around the early 2010s, discriminative correlation filter trackers
became one of the dominant families in visual tracking. Methods such as MOSSE \[5\], CSK
\[6\], KCF \[7\], and DSST \[8\] learned online filters that could localize the target
efficiently in the frequency domain, while DSST also introduced a separate scale
estimation mechanism. Later, more advanced correlation-filter trackers such as ECO \[9\]
improved robustness by using more compact model updates and stronger feature
representations. This period is commonly treated as the correlation-filter era of SOT,
before deep Siamese and transformer-based trackers became the dominant direction.

*Table 1.1 - Common tracking challenges.*

| **Challenge**      | **Example**                                      | **Why it is difficult**                                            |
|--------------------|--------------------------------------------------|--------------------------------------------------------------------|
| Occlusion          | Target is partially hidden                       | The visible evidence becomes incomplete and the tracker may drift. |
| Scale change       | Target approaches the camera                     | The size of the target box changes over time.                      |
| Deformation        | Person, animal, or flexible object changes shape | Appearance no longer matches the initial template exactly.         |
| Background clutter | Similar objects in the scene                     | The tracker can confuse target and distractor.                     |
| Motion blur        | Fast motion or camera movement                   | Fine details disappear from the search crop.                       |
| Out-of-view motion | Target leaves and returns                        | The tracker state may be corrupted during absence.                 |

## 1.2 From Siamese trackers to transformer trackers

The state of the art in single-object tracking has largely focused on template-to-target
matching, where the tracker relies on visual cues from two image regions: a template
crop that describes the target appearance and a search crop in which the target has to
be localized. A major step in this direction was introduced by SiamFC \[10\], which
formulated tracking as a fully convolutional Siamese matching problem between an
exemplar image and a larger search image. Later, SiamRPN \[11\] extended this idea by
combining Siamese feature extraction with a region proposal subnetwork, allowing the
tracker to perform target classification and bounding-box regression. In these early
Siamese trackers, the template and search images are processed by two branches with
shared convolutional weights, after which the extracted features are compared or fused
to estimate the target location.

Later trackers introduced transformer-based components for richer relation modeling
between the target and the search region. STARK \[12\] used an encoder-decoder
transformer to model global spatio-temporal dependencies and to directly predict the
target bounding box. OSTrack \[13\] further simplified the tracking pipeline by
introducing a one-stream transformer framework. Instead of extracting template and
search features independently and then applying a separate relation module, OSTrack
jointly processes template and search tokens inside the same backbone, so that feature
extraction and relation modeling are unified. Recent high-performing trackers on GOT-10k
include temporal and autoregressive models such as ODTrack \[15\], ARTrackV2 \[16\], and
ARPTrack \[17\]. These models indicate that recent tracking research increasingly
explores stronger temporal modeling, token propagation, and sequence-level pretraining.

All of these models share a similar high-level tracking structure. They accept a
template crop, which represents how the target looks, and a search crop, which contains
the region of the current frame where the target is expected to appear. The tracker then
extracts visual features from these two inputs and estimates the target position by
comparing, fusing, or jointly modeling the template and search representations. The main
architectural differences lie in how these features are extracted and how the relation
between the template and search region is modeled: earlier Siamese trackers rely mainly
on convolution and cross-correlation, while transformer-based trackers use attention
mechanisms to allow richer interaction between template and search features.

It is also important to note that feature extraction in modern trackers is often based
on pretrained backbones. These backbones are trained on large-scale image or video
datasets before being adapted to tracking. This is important because tracking datasets
alone are usually not sufficient for learning strong low-level and semantic visual
features from scratch. In practice, the pretrained backbone provides general visual
representations, while tracker training teaches the model how to use these
representations for target localization and target-background discrimination.

Another important family of trackers is represented by DiMP \[18\]. Unlike purely
Siamese approaches that mainly compare a fixed template with the current search region,
DiMP learns a discriminative target model during tracking. Its main idea is to predict a
target-specific classifier that can distinguish the target from the background using
information available from the video sequence. This makes DiMP especially relevant to
this thesis because it explicitly uses sequence information to adapt the target model
online, instead of relying only on a static initial template. The later PrDiMP \[19\]
extends this direction with a probabilistic regression formulation for visual tracking.
Although DiMP-style trackers are architecturally different from OSTrack, they show that
online adaptation and sequence-level information can improve tracking robustness. This
motivates the idea of adding recurrent memory to a template-search transformer tracker.

## 1.3 Template-search formulation

The most common notation in modern single-object tracking uses a template crop and a
search crop. The template crop is usually denoted by z and is extracted from the first
annotated frame. It contains the target object and a limited amount of surrounding
context. The search crop is usually denoted by x_t and is extracted from the current
frame around the expected target location. The tracker predicts the target box inside
this search crop rather than inside the full image.

Search crops are used instead of full frames for both statistical and computational
reasons. Statistically, the previous target position gives a useful prior: in short-term
tracking the target is usually expected near its previous location, so there is no need
to process the whole image at full resolution. Computationally, transformer-style
attention is quadratic in the number of tokens \[21\]. If a 1920 by 1080 image is split
into 16 by 16 patches, it contains roughly 120 by 68, or about 8160, tokens. A 256 by
256 search crop with the same patch size contains only 16 by 16, or 256, tokens. The
attention matrix for the full frame would have about 8160^2, or 66.6 million, pairwise
interactions, while the search crop has 256^2, or 65 thousand. This is about a thousand
times smaller before considering multiple layers, heads, and batch elements.

For this reason, both the template and the search region are usually cropped and resized
to fixed network input sizes. In this project, the template size is 128 and the search
size is 256. Resizing makes batching possible and allows the tracker to use a fixed
transformer architecture. It also means that all box predictions must be interpreted in
the coordinate system of the resized crop and then converted back to original image
coordinates.

In many trackers the template is cropped only at initialization and remains fixed during
the sequence. Other trackers update or maintain an additional dynamic target
representation during inference. FEAR \[14\], for example, uses a dual-template
representation with dynamic template update to incorporate temporal information
efficiently. DiMP \[18\] follows a different route: instead of simply storing a new
template crop, it learns a discriminative target model from sequence information,
including background evidence. These examples show that temporal information can be
introduced at different points in the tracking pipeline. The memory mechanism in this
thesis follows yet another route: it adds latent recurrent tokens directly inside the
transformer token sequence.

## 1.4 Evaluation metrics and common benchmarks

The most important localization measure in this thesis is intersection over union (IoU),
also called overlap. For a predicted box and a ground-truth box, IoU is the area of
their intersection divided by the area of their union. A value of 1 means perfect
overlap, while a value of 0 means that the boxes do not overlap.

*IoU(B\_{pred}, B\_{gt}) = \|B\_{pred} ∩ B\_{gt}\| / \|B\_{pred} ∪ B\_{gt}\|.*

GOT-10k commonly reports Average Overlap (AO), SR0.50, and SR0.75 \[20\]. AO is the
average overlap over evaluated frames. SR0.50 is the proportion of frames whose overlap
is above 0.50, and SR0.75 is the proportion of frames whose overlap is above 0.75.
SR0.75 is stricter and is more sensitive to precise localization errors. These metrics
are useful together because a tracker may have reasonable coarse localization but lower
precise localization.

GOT-10k is a suitable benchmark for this thesis because it was designed for generic
object tracking and uses a one-shot protocol with separated training and testing object
classes \[20\]. This makes it harder to overfit to a fixed set of categories. A model
must learn a general tracking rule rather than a closed-set detector. The experiments in
this thesis use exactly these metrics and compare the proposed tracker with reported
OSTrack-256 + CE and OSTrack-384 + CE results.

## 1.5 Transformers, GRUs, and backpropagation through time

This section is included because the proposed architecture combines two modeling ideas
that are usually discussed separately. Transformers model relationships among tokens
through attention. A vision transformer splits an image into patches, embeds those
patches as tokens, and processes the token sequence with transformer blocks \[22\].
OSTrack builds on this idea by treating template and search image patches as tokens and
processing them jointly.

Recurrent neural networks provide a different kind of sequence modeling. A GRU is a
gated recurrent unit designed to control how much previous state is kept and how much
new input is accepted \[23\]. The GRU is simpler than a long short-term memory unit but
still includes gates that regulate information flow. For tracking, this is appealing
because a memory representation should not be replaced blindly at every frame. Some
target information should be preserved, while outdated or corrupted information should
be discarded.

Training a recurrent tracker also introduces backpropagation through time (BPTT). If the
tracker is unrolled over a sequence of frames, the loss at a later frame depends on the
memory states produced at earlier frames. Gradients therefore flow backward not only
through layers, but also through time. Full BPTT over an entire video can be expensive
and unstable, so practical training often uses a fixed sampled sequence length. In this
project, the initial training setup used rollouts of 20 search frames, which can be
interpreted as a truncated BPTT setting: the model learns from temporal dependencies
inside the sampled window, while memory is reset at sequence boundaries. In later
training stages, the rollout length was reduced to 15 search frames to lower the memory
and optimization cost of recurrent training.

The combination used in this thesis is therefore experimental but natural. A transformer
backbone performs token-level feature interaction, while a GRU-based module updates
explicit memory tokens across frames and layers. The design is motivated by the fact
that tracking is inherently sequential, while the selected baseline is a transformer
model that already represents images as token sequences.

## 1.6 Motivation for temporal memory

A single image is sometimes not enough to recognize or localize an object reliably. In
video, motion can reveal structure that is almost invisible in a static frame. A
camouflaged animal may be hard to distinguish from the background until it moves. A thin
object, such as a snake in grass, may be visually ambiguous in one frame but becomes
recognizable when its position changes coherently over several frames. A target that is
partly hidden in the current frame may still be identifiable because it was clearly
visible a few frames earlier.

The initial template is a strong identity cue, but it captures only one appearance of
the target. During a video, the target may rotate, become brighter or darker, change
scale, deform, or become partially occluded. A memory stream can store evidence from
later frames and may represent recent target appearance better than the initial template
alone. This is the intuitive reason to add temporal memory to a tracker.

A naive solution would be to pass all previous frame tokens to the transformer together
with the current search crop. This is usually impractical. If one search crop contains
256 tokens, then 20 frames contain 5120 tokens before adding template tokens. Attention
over 5120 tokens requires about 26 million pairwise token interactions per layer. A
single-frame search crop requires only about 65 thousand interactions. Therefore, full
video-token attention over even a short sequence is roughly 400 times larger than one
search crop. If full-resolution frames were used, the cost would be much higher.

Memory tokens offer a compact alternative. Instead of storing every patch token from
every previous frame, the model keeps a small learned latent state. This state is
updated after each frame and passed forward. In principle, it can summarize useful
information while discarding irrelevant details. The challenge is that this summary must
be learned. If the memory update is too aggressive, the tracker may store distractor
information after a wrong prediction. If it is too conservative, the memory will not
adapt. The two-stage GRU used in this thesis is an attempt to update memory through
gates rather than direct replacement.

# 2. OSTrack Baseline

## 2.1 One-stream transformer tracking

OSTrack is a one-stream transformer tracker. The term one-stream means that template and
search tokens are processed together in a single backbone instead of being encoded
separately and then matched by a separate relation module. The OSTrack paper argues that
joint processing allows feature learning and relation modeling to be unified \[13\].
This is an important distinction from many two-stream Siamese approaches.

In a simplified notation, let T\_{t,l} denote template tokens and S\_{t,l} denote search
tokens at layer l for frame t. In the baseline without memory tokens, a transformer
layer processes their concatenation.

*\[S\_{t,l}, T\_{t,l}\] = ViTLayer_l(\[S\_{t,l-1}, T\_{t,l-1}\]).*

The search tokens after the final backbone layer are passed to a prediction head. The
head produces classification and localization information, ultimately yielding the
target bounding box. The exact head implementation is not the main focus of this thesis.
What matters is that the baseline has no explicit recurrent state that is propagated
across frames inside the backbone.

The one-stream structure is efficient and strong because template-search interaction is
not delayed until after independent feature extraction. Each transformer layer can
propagate information between the template and search tokens. This also creates a
challenge for memory research: because the baseline already has strong target-search
interaction, a new memory pathway may be ignored unless the training setup makes
temporal information useful.

## 2.2 ViT backbone used in this thesis

OSTrack is built on a Vision Transformer backbone. In this thesis, most architectural
changes focus on augmenting the standard ViT-based OSTrack backbone rather than
replacing it. ViT is a transformer-based feature extraction model for images. Instead of
using convolutional filters, it divides an input image into fixed-size patches, converts
each patch into a token embedding, adds positional information, and processes the
resulting sequence with transformer encoder layers. This allows the model to learn
visual representations through self-attention, where each image patch can interact with
other patches in the same token sequence. ViT was introduced by Dosovitskiy et al. and
has since become a strong alternative to convolutional backbones in many vision tasks
\[22\].

OSTrack adapts this ViT structure to visual object tracking. The template crop and the
search crop are both converted into patch tokens and then processed together by the same
transformer backbone. This is important because the backbone does not only extract
visual features; it also models the relationship between the target template and the
current search region. In this way, OSTrack unifies feature extraction and
template-search interaction inside one transformer stream \[13\].

The evaluated MemOSTrack configuration follows the OSTrack-256 + CE backbone setting.
It uses the `vit_base_patch16_224_ce` backbone with a 128 x 128 pixel template crop and
a 256 x 256 pixel search crop. With the ViT-B/16 patch size, these crops correspond to
64 template tokens and 256 search tokens. The 256 search resolution, ViT-B/16 backbone
family, and CE-enabled token-pruning mechanism make the experiment directly comparable
to the OSTrack-256 + CE configuration reported in the original OSTrack article.

It is also important that OSTrack uses a pretrained ViT backbone. The original OSTrack
paper shows that the choice of backbone initialization has a significant effect on
tracking performance, with pretrained ViT models performing much better than training
the backbone from scratch \[13\]. This is expected because tracking datasets are usually
not large enough to learn strong general-purpose visual features without pretraining.
In this thesis, the model uses the `mae_pretrain_vit_base.pth` MAE-pretrained ViT-Base
checkpoint, following the OSTrack configuration. MAE pretraining teaches the backbone
useful visual representations by masking image patches and training the model to
reconstruct the missing content \[24\]. This makes the backbone a strong starting point
for tracking, while the proposed memory extension can be studied as an augmentation of
the existing OSTrack architecture rather than as a completely new feature extractor.

## 2.3 Candidate elimination

OSTrack introduces candidate elimination (CE) to reduce the number of search tokens
processed by the transformer backbone. In a template-search tracker, the search crop
contains both the target and many background patches. CE removes search tokens that are
unlikely to correspond to the target, reducing computation while keeping the most
relevant candidates [13].

OSTrack chooses candidates using attention between the template and search tokens. After
a self-attention layer, a representative template token is used to score the search
tokens. In the standard setting, this representative token is the center template token,
because the target is normally centered in the template crop. If this token is denoted
by $\phi$ and the transformer has $M$ attention heads, the score of search token $x$ is
computed by averaging attention from $\phi$ to $x$ over all heads:

$$
w_x^{\phi} = \frac{1}{M}\sum_{m=1}^{M} w_x^{\phi}(m).
$$

Search tokens with the highest scores are kept, while lower-scoring tokens are removed
from the sequence for later transformer layers. If $k$ tokens are kept from $n$ current
search tokens, the keep ratio is:

$$
\rho = \frac{k}{n}.
$$

The kept search-token indices can be written as:

$$
I_{\text{keep}} = \operatorname{TopK}(w_x^{\phi}, k).
$$

The original positions of the kept search tokens are stored so that the spatial search
feature map can be reconstructed before the prediction head [13].

This matters for MemOSTrack because the proposed model adds memory tokens to the
OSTrack token sequence. In this implementation, CE is applied only to search tokens.
Template tokens are preserved, and memory tokens are also preserved because they are
recurrent latent states rather than spatial search candidates. Memory tokens participate
in attention, but they are not used to score search candidates and are not removed by CE.
Thus, candidate elimination keeps its original role of pruning search-region candidates,
while the recurrent memory stream remains available throughout the backbone.

## 2.4 Standard OSTrack training

Standard pair-based tracking training samples a template frame and a search frame from a
video. The template crop is centered around the target in the template frame, and the
search crop is centered around the target in the search frame. The model is trained to
predict the target box in the search crop. This is appropriate for a memory-free tracker
because each training sample can be treated independently.

*sample = (z, x_t, B_t), \hat{B}\_t = f(z, x_t).*

Here z is the template crop, x_t is the search crop, and B_t is the target box in the
search crop. The model predicts \hat{B}\_t, and the loss compares it with B_t. This
setup does not require any state from previous search frames.

## 2.5 Limitations of pair-based tracking for memory

Pair-based training has a mismatch with recurrent memory. If the model has memory state
M_t, then the value of this state should come from previous frames in the same video. A
random template-search pair does not provide a meaningful previous state. Resetting
memory for every pair would make the memory branch nearly useless, while carrying memory
across unrelated samples would be incorrect.

Pair-based sampling also hides some inference difficulties. In training, the search crop
is often centered using the ground-truth box of the current search frame. In inference,
however, the tracker does not know the current ground truth. It crops the next search
region using the previous predicted location. This means that errors can accumulate. A
model trained only on perfect ground-truth-centered crops may be less prepared for
drift.

These two limitations motivated the major training-pipeline changes described in Chapter 4.
The architecture alone was not enough. A recurrent tracker also needed a recurrent
training process.

# 3. Proposed Memory-Augmented OSTrack

## 3.1 Motivation and controlled scope

The proposed model adds explicit memory tokens to OSTrack. The motivation is
that tracking is not only a matching problem between the first template and the
current search crop. It is also a temporal problem. The target observed in
recent frames may contain information that is absent from the initial template,
especially after viewpoint changes, deformation, illumination changes, or
partial occlusion.

Memory tokens are a natural extension for a transformer tracker because OSTrack
already represents the template and search images as token sequences. The memory
tokens are not image patches. They are latent vectors with the same embedding
dimension as the ordinary transformer tokens, so they can be concatenated with
search and template tokens and processed by the same attention layers.

The implementation deliberately keeps the scope controlled. The patch embedding,
main OSTrack backbone structure, and main prediction head are kept close to the
baseline. This is important for interpretation. If the backbone, prediction
head, loss, sampler, and memory mechanism were all replaced at the same time,
it would be difficult to know which change caused the final behavior.

Two architectural variants were explored. The first variant adds memory tokens
without an explicit recurrent gate. This tests the minimal change needed to add
a memory stream to OSTrack. The second variant adds a GRU-based update, which
was introduced after the first variant revealed that direct memory replacement
is not controlled enough for a recurrent tracker.

## 3.2 Variant A: Direct memory update by the ViT layer

The first experiment was the simplest memory-augmented version of OSTrack. In
this variant, memory tokens are added to the transformer token sequence and are
updated directly by the same ViT layer that processes the search and template
tokens. The purpose of this variant was to test whether ordinary
self-attention, residual connections, and feed-forward transformations are
already sufficient to maintain a useful memory state.

At frame `t` and layer `l`, the token sequence is written as:

$$
Y_{t,l} = \left[ S_{t,l}, T_{t,l}, M_{t,l} \right].
$$

Here, `S_{t,l}` denotes the search tokens, `T_{t,l}` denotes the template
tokens, and `M_{t,l}` denotes the memory tokens after layer `l` while processing
frame `t`. All three token groups have the same embedding dimension, so they can
be concatenated along the token dimension and passed through the same
transformer block.

The main token shapes used in the architecture are:

| **Symbol** | **Meaning**    | **Shape**               |
|------------|----------------|-------------------------|
| $S_{t,l}$  | Search tokens  | $B \times N_s \times D$ |
| $T_{t,l}$  | Template tokens | $B \times N_t \times D$ |
| $M_{t,l}$  | Memory tokens  | $B \times N_m \times D$ |

Here, $B$ is batch size, $N_s$ is the number of search tokens, $N_t$ is the number of
template tokens, $N_m$ is the number of memory tokens, and $D$ is the token dimension.

In the direct-update variant, the ViT layer itself produces the next search,
template, and memory tokens:

$$
\left[ S_{t,l}, T_{t,l}, M_{t,l} \right]
=
\operatorname{ViTLayer}_l
\left(
\left[ S_{t,l-1}, T_{t,l-1}, M_{t,l-1} \right]
\right).
$$

In this equation, `M_{t,l-1}` is the memory input from the previous transformer
layer in the current frame, and `M_{t,l}` is the memory output produced directly
by the ViT layer. After the frame is processed, the resulting layer-wise memory
states are stored and used as the previous-frame memory when the next video
frame is processed. This gives the model a recurrent path through the video, but
the update itself is still only the ordinary transformer output. There is no
explicit gate that decides how much old memory should be preserved.

This variant is useful because it isolates the effect of adding memory tokens
and letting the existing transformer machinery update them. It does not
introduce a separate recurrent update rule. If this variant had worked well, the
conclusion would have been that self-attention over memory tokens is sufficient
to create a useful recurrent state.

## 3.3 Limitation of direct memory replacement

The first variant exposed two separate limitations. The first limitation is the
lack of a dedicated memory gate. A ViT block updates tokens through
self-attention, residual connections, normalization, and a feed-forward network.
This is a powerful feature transformation, but it is not the same as a
recurrent memory update. In the implementation, memory tokens are placed before
the visual tokens when a transformer block is called. If `V_{t,l-1}` denotes the
ordinary visual tokens after template-search concatenation, the input to layer
`l` is:

$$
X_{t,l-1} = \left[ M^{in}_{t,l-1}, V_{t,l-1} \right].
$$

The ViT block itself follows the pre-normalization residual form used in
`vit.py`:

$$
\begin{aligned}
U_{t,l}
&=
X_{t,l-1}

+ \operatorname{DropPath}_l(
  \operatorname{Attn}_l(
  \operatorname{LN}_{1,l}(X_{t,l-1})
  )
  ), \\
  X_{t,l}
  &=
  U_{t,l}
+ \operatorname{DropPath}_l(
  \operatorname{MLP}_l(
  \operatorname{LN}_{2,l}(U_{t,l})
  )
  ).
  \end{aligned}
  $$

After this block, the first `K` tokens are split out as the memory candidate:

$$
\left[ \hat{M}_{t,l}, V_{t,l} \right]
= \operatorname{split}(X_{t,l}).
$$

In the direct no-GRU path, this candidate is normalized and carried forward:

$$
M_{t,l} = \operatorname{LN}^{mem}_l(\hat{M}_{t,l}).
$$

These equations show why the update is not a recurrent gate. The attention
weights decide which tokens interact inside the current layer, and the residual
path helps preserve the current input to that layer. However, there is no
explicit variable that says: keep the old memory, accept the new observation, or
blend the two in a controlled proportion. The memory token produced by the
transformer is therefore best understood as a candidate feature, not as a safely
updated recurrent state.

This matters because the current search crop is not always trustworthy. The
target may be occluded, motion-blurred, visually close to a distractor, or
shifted away from the crop center after a previous prediction error. In the
direct variant, a corrupted observation can enter the memory tokens through
ordinary attention and then be carried into later frames. The transformer can
learn to reduce this risk implicitly, but it has to discover that behavior as
part of the general token-mixing function. For a recurrent tracker, this is too
weak: memory preservation should be an explicit operation, not only an emergent
side effect of attention.

The second limitation is the lack of same-depth temporal skip connections. In
the direct variant, memory is propagated mainly by feeding the previous memory
tokens into the next transformer computation. Once the current frame starts,
higher-layer memory is produced through the vertical layer-by-layer path. This
means that memory at layer `l` does not have a direct gated connection to the
memory from the previous frame at the same layer, `M_{t-1,l}`. Temporal
information must pass through a longer chain of transformer operations before it
can influence deeper memory states. This makes the recurrent path less direct
and gives losses from later frames a longer and less controlled route back to
the memory states that produced earlier predictions.

The second architectural variant was introduced to address both problems. The
transformer output is treated only as candidate memory, `M'_{t,l}`. A temporal
GRU then fuses this candidate with the previous-frame memory from the same
layer, creating an explicit time skip connection. A second GRU fuses the result
with the previous-layer memory in the current frame. This makes memory updating
more intentional: the model still uses attention to propose new information,
but recurrent gates decide how much of that information should enter the final
memory state.

## 3.4 Variant B: GRU-based memory update and two-stage architecture

The second experiment adds a gated recurrent update to the memory stream. In
this version, the transformer layer does not directly produce the final memory.
Instead, it produces a candidate memory value:

```math
[S_{t,l}, T_{t,l}, M'_{t,l}] =
\mathrm{ViTLayer}_l([S_{t,l-1}, T_{t,l-1}, M_{t,l-1}]).
```

The value `M'_{t,l}` is the memory suggested by attention at the current frame
and layer. It is not accepted directly. It is passed to a two-stage GRU update.
The first GRU combines candidate memory with memory from the previous frame at
the same layer:

```math
\tilde{M}_{t,l} = \mathrm{GRU}_1(M'_{t,l}, M_{t-1,l}).
```

The second GRU combines the intermediate memory with memory from the previous
layer in the current frame:

```math
M_{t,l} = \mathrm{GRU}_2(\tilde{M}_{t,l}, M_{t,l-1}).
```

The final output of the layer is:

```math
Y_{t,l} = [S_{t,l}, T_{t,l}, M_{t,l}].
```

The first GRU can be interpreted as temporal fusion. It decides how much memory
from frame `t-1` should remain when the current candidate memory is introduced.
The second GRU can be interpreted as layer-wise fusion. It allows memory from
the previous transformer layer to influence the final memory at the current
layer, creating a gated depth path for memory tokens.

This design was added late in development, after the simpler memory-token
variant showed that merely appending memory tokens does not provide a controlled
update. The GRU update makes the memory stream explicitly recurrent and gives
the model a learned mechanism for balancing preservation and adaptation.

A related alternative considered during development was a three-way gated
update. In that design, the final memory would be a weighted combination of the
three available memory sources:

```math
M_{t,l} = \alpha_1 M'_{t,l}
        + \alpha_2 M_{t-1,l}
        + \alpha_3 M_{t,l-1},
\quad
\alpha_1 + \alpha_2 + \alpha_3 = 1.
```

Such a softmax-style gate would make the three-source structure explicit. The
implemented version instead uses two sequential GRU updates. The GRU version is
more recurrent in form and uses learned gates inside each update. The three-way
gated update remains a useful future ablation because it would test whether a
simpler explicit fusion rule is sufficient.

# 4. Sequence-Based and Inference-Like Training

## 4.1 Why the original sampler was insufficient

The original training sampler was designed for a memory-free template-search tracker. It
could sample a template frame and a search frame independently from a video and create a
valid training pair. This is not enough for MemOSTrack. The memory state at frame t
should depend on the preceding frames. If the training sample contains only one search
frame, there is no realistic sequence through which memory can evolve.

The first required rewrite was therefore to make the sampler return video frames as a
sequence. This changed the training object from a single pair into an ordered set of
frames. The model could then be executed repeatedly, carrying memory from one search
frame to the next.

*sample = (z, x_1, x_2, ..., x_N, B_1, B_2, ..., B_N).*

For each search frame, the model predicts a box and updates memory. The important point
is that the order is preserved. Frame t must be processed after frame t-1, not as an
independent shuffled sample.

## 4.2 First rewrite: sequence sampling

The first sampler rewrite introduced causal consecutive sampling. In the project
configuration, the sampler mode was set to a causal consecutive mode, and the number of
search frames was set to 20. The intention was to build sequences such as F_t, F\_{t+1},
..., F\_{t+19}.

The word causal is important. A tracker at time t should not use future frames to
determine its current memory state. The sequence is processed in the same direction as
inference: previous frames influence later frames, not the reverse. This is different
from video understanding tasks where a model may be allowed to look both forward and
backward in time.

This rewrite made it possible to train the GRU update. After each frame, the memory
output becomes the memory input for the next frame. Losses can be applied at each step
or accumulated over the sequence.

Sequence sampling solved the temporal ordering problem, but it did not solve the
train/inference mismatch. If every search crop is centered using the ground-truth box of
that same frame, training remains artificially easy. The target is always near the
expected location, and the model does not experience the consequences of its own
previous prediction errors.

During inference, the search crop for frame t is produced using the previous predicted
box:

*x_t = crop(I_t, \hat{B}\_{t-1}).*

Training with ground-truth-centered crops instead uses:

*x_t = crop(I_t, B^{gt}\_t).*

These are different. The second equation gives the tracker information that is not
available at inference time. The difference matters more for recurrent memory because
memory can amplify errors. If the tracker is never trained under realistic drift, the
memory may behave poorly when drift occurs during testing.

## 4.3 Dynamic search cropping

The final sampling rewrite introduced dynamic search crop generation. Instead of
precomputing all search crops around their ground-truth boxes, the training loop
generates each next crop from the current tracker state. The state is initialized with
the ground-truth target box in the first frame, but after that the model's own
predictions affect future crops.

The training loop becomes: initialize the tracker state from the first-frame ground
truth; crop the current search frame around the current state; run the model on the
template, search crop, and current memory; predict the target box; update the recurrent
memory; convert the prediction back to image coordinates; and use that prediction to
crop the next frame.

*c_t = crop(I_t, \hat{B}\_{t-1}), \hat{B}\_t, M_t = f(z, c_t, M\_{t-1}).*

This loop is more difficult than ordinary pair-based training, but it is conceptually
closer to inference. The tracker sees the consequences of its own predictions, and the
memory state is updated under conditions similar to deployment.

## 4.4 Coordinate transformations in dynamic cropping

Dynamic cropping requires careful coordinate handling. The tracker predicts a box in the
coordinate system of the search crop, but the next crop must be produced in the
coordinate system of the original image. Therefore, each crop operation must store the
mapping between image coordinates and crop coordinates. If this mapping is inconsistent,
the next search crop will be centered incorrectly even when the prediction inside the
current crop is accurate.

The practical pipeline contains three coordinate spaces. The first is the original image
space, where dataset annotations are defined. The second is the raw crop space, where a
rectangular region is extracted around the current tracker state. The third is the
resized network input space, for example 256 by 256 pixels for the search image. The
model prediction is produced in the network input space and must be mapped back through
the resize operation and crop offset.

This is one of the reasons why dynamic cropping is more complex than static ground-truth
cropping. With static cropping, the crop can be generated from known annotations before
the forward pass. With dynamic cropping, the crop depends on a prediction that is only
available after the forward pass. The training loop must therefore interleave model
execution, loss computation, coordinate conversion, and crop generation.

*\hat{B}^{image}\_t = transform^{-1}(\hat{B}^{crop}\_t).*

This mapping must also be considered when computing losses. The ground-truth box can be
transformed into the crop coordinate system, or the prediction can be transformed into
the image coordinate system. Both choices are valid if implemented consistently. The
thesis does not claim that coordinate handling is a new scientific contribution, but it
is a necessary engineering part of making the proposed recurrent training pipeline work.

## 4.5 Backpropagation through time in the training loop

Once the model is executed over a sequence, training is no longer a collection of
independent frame pairs. The memory state connects the frames. If the loss is
accumulated over a sequence, then the prediction at frame t can influence not only the
current loss, but also later losses through the memory state and through dynamic
cropping. Backpropagation through time is the mechanism that propagates these gradients
backward along the unrolled sequence.

For a sequence of N search frames, the main tracking loss can be written as:

*L\_{seq} = \sum\_{t=1}^{N} L\_{track}(\hat{B}\_t, B^{gt}\_t).*

The exact tracking loss depends on the OSTrack implementation, but it usually includes
localization terms and classification-like terms. The important design issue is that the
loss is not only a single-frame loss. It supervises the model while memory is being
propagated. In practice, using a fixed sequence length such as 20 frames is a form of
truncated BPTT, because gradients are propagated through the sampled window rather than
through the full video.

This has practical consequences. Longer sequences expose the memory to more temporal
behavior, but they increase GPU memory usage and make optimization more difficult.
Shorter sequences are easier to train, but they may not teach the memory enough about
long-term appearance change. The sequence length used in this project is therefore a
compromise between recurrent learning and computational feasibility.

## 4.6 Curriculum considerations

An inference-like crop pipeline may be introduced gradually. One possible curriculum is
to begin training with ground-truth-centered sequence crops so the memory update learns
a stable target representation. After a warm-up phase, dynamic cropping can be enabled
so the model learns to handle prediction drift. A second option is mixed cropping: with
some probability the crop is centered on ground truth, and with the remaining
probability it is centered on the previous prediction. This can reduce early instability
while still exposing the tracker to realistic errors.

The model evaluated in this thesis should therefore be interpreted as one point in a
larger design space. The final training pipeline is realistic, but it may not be the
easiest optimization path. If the memory module underperforms, it is not enough to
conclude that memory is useless. It may be necessary to design a curriculum in which
memory first learns from clean sequences and later learns to survive drift.

This observation is consistent with broader sequence-level tracking work, which argues
that frame-level training can mismatch the test-time objective of maintaining
localization quality over a whole sequence \[25\]. MemOSTrack addresses the same
general issue from a different angle: it rewrites the sampler and crop generation so
that recurrent memory is trained in a sequential setting.

## 4.7 Why inference-like training is harder

Dynamic cropping makes training harder. With ground-truth-centered crops, the model sees
clean examples even if its previous predictions would have been inaccurate. With dynamic
cropping, an early localization error can move the next crop away from the target
center. The target may appear near the edge of the crop, partially outside the crop, or
surrounded by more distractors. This can reduce training stability.

However, this difficulty is intentional because it makes training closer to inference.
A tracker deployed in inference does not receive perfect search crops. If the goal is to
train a recurrent memory module, the memory should be exposed to realistic inputs.
Otherwise, the memory may learn to work only in an idealized setting where every frame
is perfectly centered.

This trade-off is important for interpreting the final results. MemOSTrack improves over
the same-resolution OSTrack-256 + CE baseline, but remains below the higher-resolution
OSTrack-384 + CE baseline. The more demanding training pipeline may have affected
optimization, but this cannot be isolated without ablation experiments.

# 5. Auxiliary Memory Supervision

## 5.1 Auxiliary Loss Concept

Adding memory tokens to a strong model does not guarantee that the model will use them.
OSTrack already has a powerful template-search pathway. Since the tracker can solve the
training task using template and search tokens alone, the new memory tokens may receive
weak or unimportant gradients. This creates the risk that the memory tokens collapse to an uninformative representation
or are ignored by the prediction pathway.

This is a common issue when adding auxiliary modules to strong neural networks. The
optimization process often finds the easiest path to reduce the loss. If the template is
clean and the search crop is well-centered, the easiest path may be to rely on the
original OSTrack features. The memory branch may then have little effect on the final prediction.

The project therefore explored auxiliary memory supervision. The goal was to make memory
tokens directly useful for prediction rather than merely present in the token sequence.

## 5.2 Direct Memory Filter Loss

The first auxiliary loss explored in this work was the **Direct Memory Filter Loss**. Its
purpose was to make the memory branch produce filters that are directly useful for target
localization. For a predicted memory filter $f_t$, a sampled search feature $x_j$, and a
ground-truth heatmap $y_j$, the filter response is:

$$
s_{t,j} = x_j * f_t.
$$

The auxiliary loss compares this response with the target heatmap and adds a small filter
regularization term:

$$
L_{\text{direct}} = \operatorname{mean}\left(\lVert s_{t,j} - y_j \rVert_2^2\right)
+ \lambda \operatorname{mean}\left(\lVert f_t \rVert_2^2\right).
$$

Here, $\lambda$ corresponds to `FILTER_REG`. In the implementation, a limited number of
search frames is sampled using `FILTER_LOSS_FRAMES`. The search features are detached on
purpose, so the auxiliary gradient is directed mainly through the memory tokens and the
memory-filter prediction path rather than through the backbone.

The strength of this loss is that it tests the memory filter in a direct tracking-like
way: the filter is applied to search features and should produce a high response near
the target. However, the method has two important limitations. First, supervision depends
on sampled frames, so the signal can be noisy. Second, applying every predicted filter to
every previous frame would require approximately $O(T^2)$ filter-feature applications for
a sequence of length $T$. This motivated the more compact DiMP-style target-match loss.

## 5.3 DiMP Target-Match Loss

The second auxiliary loss is the DiMP-style target-match loss. Instead of directly testing every predicted memory
filter on sampled frames, this loss first constructs a stronger sequence-level target filter and then trains each
predicted memory filter to match it.

This idea is inspired by DiMP, where tracking is formulated as learning a target-specific discriminative model from
training samples. DiMP shows that online target model prediction can use both target and background information,
instead of only matching a target template. ([arXiv][2])

The proposed DiMP-style target-match loss is not a reimplementation of the full DiMP tracker. Instead, it borrows the
idea of fitting a discriminative target filter from a sequence of search features. The fitted filter is used as a
detached pseudo-label for the memory branch. This differs from original DiMP, where the optimized filter is the actual
target classifier used for tracking. The simplification is intentional: the goal is not to replace OSTrack's prediction
head, but to provide direct supervision that encourages the added memory tokens to encode information useful for target
discrimination.

The implementation first collects all available detached search features:

$$
x_1, \dots, x_T
$$

and their ground-truth heatmaps:

$$
y_1, \dots, y_T.
$$

It then fits a target filter $f^*$ by minimizing:

$$
L(f) = \operatorname{mean}\left(\lVert x * f - y \rVert_2^2\right)
+ \lambda \operatorname{mean}\left(\lVert f \rVert_2^2\right).
$$

Here, $f^*$ is the filter that best matches the whole training sequence under this least-squares objective. The code
solves this using a DiMP-style steepest-descent solver with an analytic step length. The inner solver is run under
`torch.no_grad()`, so the optimized target filter is treated as a pseudo-label rather than as a fully differentiable
inner optimization process.

After computing the target filter, the predicted memory filters are trained to match it:

$$
L_{\text{match}} = w_{\text{match}} \frac{1}{M} \sum_i \lVert f_i - f^* \rVert_2^2.
$$

where:

- $f_i$ is the predicted memory filter from frame $i$.
- $f^*$ is the optimized sequence-level target filter.
- $w_{\text{match}}$ is `FILTER_MATCH_WEIGHT`.
- $M$ is the number of predicted memory filters.

In the implementation, the initial filter is the mean of detached predicted memory filters, the target filter is
optimized over all common feature frames, and the final loss is the mean squared difference between each predicted
filter and the optimized target filter.

This loss has several advantages over the direct loss. It gives the memory branch a more stable target, because $f^*$
is estimated from the full sequence rather than from a small random sample. It also avoids the quadratic cost of
applying every predicted filter to every previous frame. With $K$ inner optimization steps and $T$ frames, the fitting
cost is closer to:

$$
O(KT)
$$

instead of:

$$
O(T^2)
$$

when $K$ is fixed and much smaller than $T$. The loss also encourages temporal consistency, because all memory filters
are pulled toward the same sequence-level target model.

The main weakness is that the method depends on the quality of the fitted target filter. If $f^*$ is inaccurate, all
predicted memory filters are encouraged to imitate a poor pseudo-target. It is also more complex than the direct loss,
because it requires an inner optimization procedure. Finally, the loss supervises filter similarity rather than directly
measuring the final tracking output, so it remains an auxiliary signal rather than a replacement for the main GIoU, L1,
and localization losses. In the training actor, the memory loss is added to the main tracking objective using the
configured memory loss weight.

[2]: https://arxiv.org/abs/1904.07220 "Learning Discriminative Model Prediction for Tracking"

## 5.4 Template Blurring

Due to the added complexity of explicit auxiliary memory losses, the final approach uses **template blurring** as a
simpler memory-supervision strategy. OSTrack is already a strong one-stream tracker, where template and search features
are jointly processed through bidirectional information flow. Therefore, the model may solve the training task without
strongly relying on the added memory tokens. ([arXiv][1]) Template blurring is intended to reduce the reliability of the
template pathway and encourage the tracker to use memory tokens as an additional source of target appearance information.

Let the original template be

$$
z
$$

and the corrupted template be

$$
\tilde{z}.
$$

Two corruption modes were considered. In the first mode, the template is blurred:

$$
\tilde{z} = \text{Blur}(z).
$$

In the second mode, the template is replaced by zeros:

$$
\tilde{z} = 0.
$$

The blur mode is less aggressive because it removes high-frequency appearance details while preserving coarse target
structure. However, it may still allow the model to recover useful information from the degraded template itself. For this
reason, zero replacement was also considered. The zero mode removes the template appearance completely, making it harder
for the model to solve the task by reconstructing or compensating for the corrupted template. This creates stronger
pressure to use the information stored in memory tokens.

The implementation is controlled by a `MemoryTemplateCorruption` module. It reads configuration values for whether
corruption is enabled, the corruption mode, the number of corrupted frames, the blur kernel size, and the number of blur
passes. The implementation supports two modes: `blur` and `zero`. In `zero` mode, the corrupted template is created with
`torch.zeros_like`. In `blur` mode, repeated average pooling is applied to the template tensor. The blur kernel size must
be an odd integer greater than one, and the number of blur passes must be positive.

A key implementation detail is that the first two search frames are never selected for template corruption. This is
controlled by

$$
\texttt{skip\_first} = 2.
$$

The selected corruption candidates therefore start only after the first two search frames. This design allows the tracker
to process several clean frames before template corruption is introduced, which is useful for establishing an initial
memory state. The module samples a fixed number of frame indices from the remaining candidate frames independently for
each batch element and stores the result in a boolean blur mask.

The method also requires sufficiently long training sequences. The implementation checks that the number of search frames
is greater than

$$
2n_{\text{blur}} + \texttt{skip\_first},
$$

where \(n_{\text{blur}}\) is the configured number of frames to corrupt. If this condition is not satisfied, validation
raises an error, or the sample is skipped during mask construction. This check avoids applying corruption in very short
sequences where there are not enough clean frames and later candidate frames.

During training, the actor builds a blur mask for the full sequence. During evaluation, the blur mask is set to all false,
so template corruption is only a training-time mechanism. For each selected frame, the corrupted template is substituted
for the clean template before the network forward pass. For unselected samples, the original template is kept. Template
blurring is also made mutually exclusive with the explicit auxiliary memory losses, which avoids mixing two different
memory-supervision mechanisms.

The main advantage of template blurring is its simplicity. It does not require an additional memory-filter objective,
a DiMP-style inner solver, or a pseudo-label for memory filters. The tracker is still trained using the standard tracking
losses, while the input corruption changes which information sources are reliable. This makes the method easier to
implement and analyze than the auxiliary losses described in the previous sections.

A second advantage is that template blurring affects the actual tracking pathway. The auxiliary filter losses supervise
intermediate filter representations, while template blurring changes the information available to the full tracker.
Improved performance on corrupted-template frames would suggest that the memory tokens provide useful target appearance
information for final prediction.

However, template blurring is still an indirect form of supervision. It does not explicitly require a memory token to
encode a specific target representation. The model may still learn to rely on other cues, such as the search crop, instead
of fully exploiting memory. Another limitation is that template blurring mainly encourages memory to store appearance
information. It does not directly provide motion supervision. Therefore, it may not fully encourage memory tokens to learn
temporal cues such as target displacement, velocity, or motion consistency across frames.

Overall, template blurring creates a controlled training condition in which the original template pathway is weakened.
Compared with explicit auxiliary losses, it is less direct but substantially simpler and does not change the main tracking
objective.

[1]: https://arxiv.org/abs/2203.11991 "Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework"


# 6. Experimental Setup

## 6.1 Dataset and protocol

The experiments are performed on the GOT-10k dataset. GOT-10k is a generic object
tracking benchmark with more than 10,000 video segments and more than 1.5 million
labeled bounding boxes [20]. The test annotations are hidden, and evaluation is
performed through the official benchmark server. The OSTrack paper reports GOT-10k
performance for several configurations, including OSTrack-256 + CE and OSTrack-384 + CE
[13]. The model in this thesis is trained exclusively on GOT-10k, without external
training samples.

## 6.2 Model configuration

For evaluation, the 256-pixel search-crop baseline with CE pruning was chosen as the closest comparison.

The model configuration can be summarized as follows: template crop size 128 x 128
pixels, search crop size 256 x 256 pixels, 64 template tokens, 256 search tokens, 64
memory tokens, ViT-style OSTrack backbone, memory tokens inside the transformer token
sequence, two-stage GRU recurrent update, candidate elimination enabled with keep ratio
0.7, causal consecutive sequence sampler, and dynamic search cropping based on previous
predictions in the final pipeline. The recurrent training was initially performed with
20-frame search rollouts; in the later stages, this was reduced to 15-frame rollouts.

## 6.3 Hyperparameter and optimizer considerations

The memory-augmented model has two different kinds of trainable parameters. The
pretrained backbone already contains general visual representations. The memory tokens,
GRU modules, and memory embeddings are new or substantially changed. It is therefore
reasonable to use different optimizer parameter groups. The backbone can be trained with
a conservative learning rate, while memory-specific parameters can receive a separately
controlled learning rate.

This distinction was important in the project because memory-specific parameters such as
backbone.mem\_ and backbone.read_mem_embed had to be detected and grouped correctly. If
they were accidentally treated as ordinary pretrained backbone parameters, they might
receive a learning rate too small for new modules. If they were treated too
aggressively, they could destabilize the backbone. The correct choice is an empirical
hyperparameter, but the optimizer must at least expose the distinction.

Gradient clipping is another relevant hyperparameter. Recurrent sequence training
accumulates losses over multiple frames and backpropagates through a memory chain.
Dynamic cropping can also create hard examples when early predictions move the crop away
from the target. These factors can lead to occasional large gradients. Gradient clipping
is therefore a practical stabilization technique, especially when newly initialized
recurrent parameters are trained together with a pretrained transformer backbone.

## 6.4 Evaluation protocol

The tracker is evaluated by producing predicted bounding boxes for the test sequences
and computing AO, SR0.50, and SR0.75. AO measures average localization overlap. SR0.50
measures the fraction of frames where the tracker achieves at least moderate overlap.
SR0.75 measures the fraction of frames where the tracker achieves high overlap.

The comparison is made against reported OSTrack-256 + CE and OSTrack-384 + CE
baselines. The 256 baseline is the closest architectural comparison because it uses the
same search resolution. The 384 baseline is included as a stronger high-resolution
reference. Because the proposed model introduces several changes simultaneously, the
final result should not be interpreted as identifying a single cause. A complete study
would include ablations, such as memory tokens without dynamic cropping, dynamic
cropping without memory, one GRU instead of two GRUs, and different memory loss
weights. Those studies are listed as future work.

## 6.5 Reproducibility considerations

The reported MemOSTrack result depends on the exact configuration file, checkpoint,
GOT-10k split, and sequence-sampling setup used during evaluation. These details are
especially important for memory-based trackers because small changes in rollout length
or crop generation can produce different training behavior.

The comparison also separates external baseline numbers from the author's own results.
The OSTrack baseline numbers are reported values from the literature \[13\]. The
MemOSTrack numbers are experimental results from this work.

# 7. Results and Discussion

## 7.1 Quantitative results

The final measured result for MemOSTrack-256 + CE is shown in Table 7.1. The model
achieved AO 0.729, SR0.50 0.823, and SR0.75 0.693. The reported OSTrack-256 + CE
baseline achieves AO 0.710, SR0.50 0.804, and SR0.75 0.682. The reported OSTrack-384 +
CE baseline achieves AO 0.737, SR0.50 0.832, and SR0.75 0.708 \[13\].

*Table 7.1 - Final quantitative comparison.*

| **Method**          | **AO** | **SR0.50** | **SR0.75** | **ΔAO** | **ΔSR0.50** | **ΔSR0.75** |
|---------------------|-------:|-----------:|-----------:|--------:|------------:|------------:|
| MemOSTrack-256 + CE |  0.729 |      0.823 |      0.693 |       - |           - |           - |
| OSTrack-256 + CE    |  0.710 |      0.804 |      0.682 |  +0.019 |      +0.019 |      +0.011 |
| OSTrack-384 + CE    |  0.737 |      0.832 |      0.708 |  -0.008 |      -0.009 |      -0.015 |

The delta columns show MemOSTrack-256 + CE minus the listed method. The final
configuration improves over the same-resolution OSTrack-256 + CE baseline on all three
metrics. The gains are 0.019 AO, 0.019 SR0.50, and 0.011 SR0.75. Because only one final
configuration is evaluated, this result should be interpreted as evidence that the
approach is promising rather than as an isolated proof of the benefit of memory.

However, the result remains below OSTrack-384 + CE. The gaps are 0.008 AO, 0.009
SR0.50, and 0.015 SR0.75. This means that the current memory-augmented 256 model is
competitive with the stronger high-resolution baseline, but it does not surpass it.
The final interpretation is therefore mixed: the final MemOSTrack configuration improves
the 256 CE setting, but increasing the search resolution to 384 still gives the best
accuracy among the compared models.

## 7.2 Training-curve analysis

The merged training log was also parsed to inspect the behaviour of the final run over
time. Figure 7.1 shows the completed training rows, one point per epoch. Figure 7.2
shows the completed validation rows, which were logged every five epochs. These curves
are not a substitute for the official GOT-10k test result, but they are useful for
understanding when the main training-stage changes occurred.

![Training metrics by epoch.](../output/logs/ostrack-vitb_256_mae_ce_32x4_got10k_ep100_train_epoch_metrics.png)

*Figure 7.1 - Training loss and overlap metrics parsed from the merged log.*

The training curve shows the expected rapid early optimization. The total training loss
falls from 2.033 at epoch 1 to 0.547 at epoch 20, while the training IoU increases from
0.639 to 0.870. Template blurring was enabled only from the 21st to the 40th epoch. In
this interval, the separate `Blur/IoU` curve is reported and increases from 0.849 at
epoch 21 to a maximum of 0.875 at epoch 39. After epoch 40, template blurring was
disabled and training switched to the inference-like rollout setting. This made the
training problem harder: the training IoU drops from 0.891 at epoch 40 to 0.869 at epoch
41, while the GIoU loss increases from 0.114 to 0.139. The curve then recovers as the
model adapts to the harder rollout regime. After epoch 59, the learning rate was
significantly decreased. This is visible as an immediate reduction in training loss from
0.390 at epoch 59 to 0.364 at epoch 60.

![Validation and test-style metrics by epoch.](../output/logs/ostrack-vitb_256_mae_ce_32x4_got10k_ep100_val_test_epoch_metrics.png)

*Figure 7.2 - Validation/test-style loss and overlap metrics parsed from the merged log.*

The validation curve follows the same broad pattern, although it is noisier because it is
measured less frequently. Validation total loss decreases from 0.810 at epoch 5 to
0.667 at epoch 20. During the blurring phase, validation is logged at epochs 25, 30, 35,
and 40, so `Blur/IoU` appears only at those validation points even though blur was active
from epoch 21. The best validation total loss is 0.442 at epoch 60. Later validation
IoU values continue to increase, reaching 0.885 at epoch 75, but the official GOT-10k
test selection used the epoch-60 checkpoint. In the external test evaluation, epoch 60
was the best checkpoint, and the outputs reported in Table 7.1 are produced from that
checkpoint. This distinction is important: the local validation curves describe training
dynamics, while the final AO, SR0.50, and SR0.75 numbers are the benchmark test outputs.

## 7.3 Token-budget interpretation

These results should also be interpreted in terms of the token budget processed by the
ViT backbone. The implemented models use a patch size of 16. Therefore, the
OSTrack-256 setting, with a 128 x 128 template and a 256 x 256 search region, produces
64 template tokens and 256 search tokens. Its initial visual sequence contains 320
tokens before candidate elimination. MemOSTrack-256 + CE uses the same visual token
budget and adds 64 learned memory tokens, so the sequence entering the first
transformer block contains 384 tokens.

The OSTrack-384 baseline uses a 192 x 192 template and a 384 x 384 search region. This
produces 144 template tokens and 576 search tokens, or 720 visual tokens before
candidate elimination. The same count can be derived exactly for the MemOSTrack
implementation if it is applied at this resolution with `MODEL.MEMORY.NUM_TOKENS = 64`:
the initial sequence would contain 144 template tokens, 576 search tokens, and 64 memory
tokens, for a total of 784 tokens. This MemOSTrack-384 count is a derived token budget,
not a trained or evaluated model result in this thesis.

*Table 7.2 - Input token counts before candidate elimination.*

| **Model setting**                | **Template tokens** | **Search tokens** | **Memory tokens** | **Initial tokens** |
|----------------------------------|--------------------:|------------------:|------------------:|-------------------:|
| OSTrack-256 + CE                 |                  64 |               256 |                 0 |                320 |
| MemOSTrack-256 + CE              |                  64 |               256 |                64 |                384 |
| OSTrack-384 + CE                 |                 144 |               576 |                 0 |                720 |
| MemOSTrack-384 + CE, derived     |                 144 |               576 |                64 |                784 |

Candidate elimination changes the active sequence length inside the backbone. In
`vit_ce.py`, CE is applied at the configured zero-based block indices 3, 6, and 9. The
implementation computes the number of retained search tokens as
`ceil(keep_ratio * current_search_tokens)`. With the configured keep ratio of 0.7 at
all three CE locations, the 256 search stream is reduced from 256 tokens to 180, then
126, then 89. The 384 search stream is reduced from 576 tokens to 404, then 283, then
199. Template tokens are not pruned. In MemOSTrack, memory tokens are prepended to the
template-side sequence for attention and are also kept across CE; however, they are
excluded from the template mask used to score search-token importance.

*Table 7.3 - Active token counts after CE pruning.*

| **Model setting**                | **Initial tokens** | **After CE 1** | **After CE 2** | **After CE 3** |
|----------------------------------|-------------------:|---------------:|---------------:|---------------:|
| OSTrack-256 + CE                 |                320 |            244 |            190 |            153 |
| MemOSTrack-256 + CE              |                384 |            308 |            254 |            217 |
| OSTrack-384 + CE                 |                720 |            548 |            427 |            343 |

This token-budget view explains why the two comparisons should be discussed
separately. MemOSTrack-256 + CE has a 20 percent larger initial sequence than
OSTrack-256 + CE because of the 64 memory tokens. At the same time, it remains much
smaller than OSTrack-384 + CE: 384 initial tokens versus 720, and 217 active tokens
versus 343 after the third CE pruning point. The remaining accuracy gap to
OSTrack-384 + CE may therefore be partly explained by the larger visual token budget
of the 384 model, rather than by the memory mechanism alone. Token count should still
be treated as an approximate proxy for computation, because the actual cost also
depends on layer depth, attention heads, CE locations, and implementation details.

## 7.4 What worked technically

In addition to the quantitative improvement over OSTrack-256 + CE, several technical
goals were achieved. The architecture was modified to include memory tokens. The memory
tokens were updated through a two-stage GRU mechanism. The sampler was rewritten to
produce ordered video sequences. The training pipeline was rewritten to use
inference-like dynamic cropping. Auxiliary memory supervision and template blurring
were studied as responses to memory underuse.

## 7.5 Limitations

The most important limitation is the lack of a full ablation study. Because multiple
changes were introduced, it is not possible to isolate the exact effect of each
component from the final table alone. For example, the gain over OSTrack-256 + CE may
come from memory tokens, dynamic cropping, candidate elimination interactions, training
differences, template blurring, auxiliary losses, or a combination of these factors.

The second limitation is that the memory mechanism was evaluated mainly on the standard
metrics. It may be more useful to analyze specific sequence attributes such as
occlusion, deformation, or long-term appearance change. If memory helps only in certain
cases, the average score may hide this benefit.

The third limitation is that memory loss heads were not fully optimized. The balance
between the main loss and memory loss is difficult and likely important. A scheduled
memory loss or a warm-up phase may be more stable than applying the full auxiliary loss
from the beginning.

## 7.6 Interpretation of the mixed result

The final result is neither a simple failure nor a general victory over OSTrack. It is
positive relative to the same-resolution OSTrack-256 + CE baseline, where MemOSTrack is
higher on all three GOT-10k metrics. At the same time, it is not enough to surpass the
OSTrack-384 + CE baseline, which benefits from a much larger search-region token budget.

The correct conclusion is therefore specific: the evaluated MemOSTrack configuration
improves the 256 CE setting, but it does not replace the accuracy benefit of the 384
search resolution. This distinction is important. It supports the memory hypothesis at
the same resolution, while the lack of ablations prevents assigning the improvement to a
single component.

## 7.7 Future work

Future work should begin with ablation experiments. The first ablation should compare
baseline OSTrack, OSTrack with sequence training but no memory, MemOSTrack with static
ground-truth-centered crops, and MemOSTrack with dynamic crops. This would separate the
effects of memory and cropping.

The second ablation should compare memory update mechanisms. A single temporal GRU, the
current two-stage GRU, and the three-way softmax gated update should be tested under the
same training conditions. This would reveal whether the layer-wise recurrent connection
is helpful.

The third ablation should vary the memory supervision strategy. Possible variants
include no memory head, memory head after selected layers only, scheduled memory loss,
and different values of `lambda_mem`. Template blurring should also be varied by blur
strength and probability.

Finally, the model should be evaluated on sequences where memory is expected to matter:
long occlusions, viewpoint changes, and gradual appearance changes. If memory helps only
on a subset of videos, attribute-level analysis may reveal benefits that average metrics
hide.

# Conclusion

This thesis investigated the integration of recurrent memory tokens into OSTrack. The
proposed MemOSTrack model extends the original one-stream transformer tracker with
memory tokens and a two-stage GRU update mechanism. The first GRU combines current
candidate memory with memory from the previous frame, while the second GRU combines the
result with memory from the previous transformer layer. This design was intended to
provide both temporal continuity and layer-wise memory refinement.

The work also required a substantial rewrite of the training pipeline. The original
pair-based sampling strategy was replaced by sequence-based sampling so that memory
could be propagated through ordered video frames. The final version further introduced
dynamic search cropping, where the search crop is generated from the previous prediction
rather than from the current ground truth. This made training more similar to inference,
although also more difficult.

Auxiliary memory loss heads were explored because a strong OSTrack baseline can ignore
newly added memory tokens. Template blurring was considered as a complementary solution
to reduce over-reliance on the clean initial template and encourage use of recurrent
memory.

The final MemOSTrack-256 + CE model achieved AO 0.729, SR0.50 0.823, and SR0.75
0.693. These results are above the reported OSTrack-256 + CE baseline of AO 0.710,
SR0.50 0.804, and SR0.75 0.682, giving improvements of 0.019 AO, 0.019 SR0.50, and
0.011 SR0.75. However, the model remains below the stronger OSTrack-384 + CE baseline
of AO 0.737, SR0.50 0.832, and SR0.75 0.708. Therefore, the final configuration improves the same-resolution 256 CE setting, but it
does not outperform the higher-resolution 384 CE model. The result provides a useful foundation for future work on memory
supervision, update mechanisms, resolution scaling, and training curricula for
transformer-based visual tracking.

# References

\[1\] D. Comaniciu, V. Ramesh, and P. Meer, 'Real-Time Tracking of Non-Rigid Objects
Using Mean Shift,' CVPR, 2000.

\[2\] B. D. Lucas and T. Kanade, 'An Iterative Image Registration Technique with an
Application to Stereo Vision,' IJCAI, 1981.

\[3\] M. Isard and A. Blake, 'CONDENSATION - Conditional Density Propagation for Visual
Tracking,' International Journal of Computer Vision, 1998.

\[4\] I. Matthews, T. Ishikawa, and S. Baker, 'The Template Update Problem,' IEEE
Transactions on Pattern Analysis and Machine Intelligence, 2004.

\[5\] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui, 'Visual Object Tracking
using Adaptive Correlation Filters,' CVPR, 2010.

\[6\] J. F. Henriques, R. Caseiro, P. Martins, and J. Batista, 'Exploiting the Circulant
Structure of Tracking-by-Detection with Kernels,' ECCV, 2012.

\[7\] J. F. Henriques, R. Caseiro, P. Martins, and J. Batista, 'High-Speed Tracking with
Kernelized Correlation Filters,' IEEE Transactions on Pattern Analysis and Machine
Intelligence, 2015.

\[8\] M. Danelljan, G. Hager, F. Shahbaz Khan, and M. Felsberg, 'Accurate Scale
Estimation for Robust Visual Tracking,' BMVC, 2014.

\[9\] M. Danelljan, G. Bhat, F. Shahbaz Khan, and M. Felsberg, 'ECO: Efficient
Convolution Operators for Tracking,' CVPR, 2017.

\[10\] L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and P. H. S. Torr,
'Fully-Convolutional Siamese Networks for Object Tracking,' ECCV Workshops, 2016.

\[11\] B. Li, J. Yan, W. Wu, Z. Zhu, and X. Hu, 'High Performance Visual Tracking with
Siamese Region Proposal Network,' CVPR, 2018.

\[12\] B. Yan, H. Peng, J. Fu, D. Wang, and H. Lu, 'Learning Spatio-Temporal Transformer
for Visual Tracking,' ICCV, 2021.

\[13\] B. Ye, H. Chang, B. Ma, S. Shan, and X. Chen, 'Joint Feature Learning and
Relation Modeling for Tracking: A One-Stream Framework,' ECCV, 2022.

\[14\] V. Borsuk, R. Vei, O. Kupyn, T. Martyniuk, I. Krashenyi, and J. Matas, 'FEAR:
Fast, Efficient, Accurate and Robust Visual Tracker,' ECCV, 2022.

\[15\] Y. Zheng, B. Zhong, Q. Liang, Z. Mo, S. Zhang, and X. Li, 'ODTrack: Online Dense
Temporal Token Learning for Visual Tracking,' AAAI, 2024.

\[16\] Y. Bai, Z. Zhao, Y. Gong, and X. Wei, 'ARTrackV2: Prompting Autoregressive
Tracker Where to Look and How to Describe,' CVPR, 2024.

\[17\] S. Liang, Y. Bai, Y. Gong, and X. Wei, 'Autoregressive Sequential Pretraining for
Visual Tracking,' CVPR, 2025.

\[18\] G. Bhat, M. Danelljan, L. Van Gool, and R. Timofte, 'Learning Discriminative
Model Prediction for Tracking,' ICCV, 2019.

\[19\] M. Danelljan, L. Van Gool, and R. Timofte, 'Probabilistic Regression for Visual
Tracking,' CVPR, 2020.

\[20\] L. Huang, X. Zhao, and K. Huang, 'GOT-10k: A Large High-Diversity Benchmark for
Generic Object Tracking in the Wild,' IEEE Transactions on Pattern Analysis and Machine
Intelligence, 2021.

\[21\] A. Vaswani et al., 'Attention Is All You Need,' NeurIPS, 2017.

\[22\] A. Dosovitskiy et al., 'An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale,' ICLR, 2021.

\[23\] K. Cho et al., 'Learning Phrase Representations using RNN Encoder-Decoder for
Statistical Machine Translation,' EMNLP, 2014.

\[24\] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, 'Masked Autoencoders
Are Scalable Vision Learners,' CVPR, 2022.

\[25\] M. Kim, S. Lee, J. Ok, B. Han, and M. Cho, 'Towards Sequence-Level Training for
Visual Tracking,' ECCV, 2022.

# Abstract

## Extending OSTrack with GRU-Based Memory Tokens for Visual Object Tracking

This thesis investigates the integration of recurrent memory tokens into OSTrack, a
one-stream transformer tracker. The proposed MemOSTrack model adds memory tokens to
the template-search token sequence and updates them with a two-stage GRU mechanism using
previous-frame and previous-layer memory. The training pipeline was rewritten from
pair-based sampling to ordered video sequence sampling and further extended with dynamic
search cropping to better match inference. Auxiliary memory loss heads and template
blurring were explored to encourage the model to use memory. The final model achieved AO
0.729, SR0.50 0.823, and SR0.75 0.693. It outperformed OSTrack-256 + CE, but remained
slightly below the stronger OSTrack-384 + CE baseline.

Keywords: visual object tracking, OSTrack, transformer, GRU, memory tokens, dynamic
cropping

# Summary

## Extending OSTrack with GRU-Based Memory Tokens for Visual Object Tracking

The thesis studies whether a transformer-based tracker can benefit from explicit
recurrent memory. OSTrack is selected as the baseline because it is a strong one-stream
tracker that jointly processes template and search tokens. The proposed MemOSTrack
model inserts memory tokens into this token sequence. A transformer layer first produces
candidate memory, after which a two-stage GRU update combines the candidate with
previous-frame memory and previous-layer memory.

A major part of the work is the training pipeline. The original pair-based sampler was
not sufficient because recurrent memory needs ordered video frames. Therefore, the
sampler was rewritten to return causal consecutive search sequences. The final pipeline
also uses dynamic search cropping, where each next search crop is produced from the
previous prediction. This makes training closer to real inference.

The thesis also discusses memory loss heads and template blurring. Memory loss heads
were introduced to prevent the model from ignoring memory tokens. Template blurring was
explored to reduce over-reliance on the clean initial template and encourage the use of
temporal memory.

The final evaluation shows that MemOSTrack-256 + CE achieved AO 0.729, SR0.50
0.823, and SR0.75 0.693. These results are higher than the reported OSTrack-256 + CE
baseline, but lower than the reported OSTrack-384 + CE baseline. The main conclusion is that the final MemOSTrack configuration improves the
same-resolution 256 CE setting, but does not yet surpass the higher-resolution 384 CE
model. Further optimization and ablation studies
are required to separate the effects of memory, dynamic cropping, candidate elimination,
and resolution.

Keywords: visual object tracking, OSTrack, memory tokens, recurrent neural network, GRU,
transformer, inference-like training

# Abbreviations

*Table A.1 - Abbreviations.*

| **Abbreviation** | **English term**                  | **Explanation**                                                 |
|------------------|-----------------------------------|-----------------------------------------------------------------|
| AO               | Average Overlap                   | Average IoU over evaluated frames                               |
| BPTT             | Backpropagation Through Time      | Training procedure for recurrent models unrolled over sequences |
| CE               | Candidate Elimination             | OSTrack efficiency module that removes unlikely tokens          |
| GOT-10k          | Generic Object Tracking benchmark | Large-scale tracking benchmark                                  |
| GRU              | Gated Recurrent Unit              | Recurrent neural network unit with gates                        |
| IoU              | Intersection over Union           | Overlap between predicted and ground-truth boxes                |
| MAE              | Masked Autoencoder                | Self-supervised vision pretraining method                       |
| SOT              | Single-Object Tracking            | Tracking one specified object through a video                   |
| SR               | Success Rate                      | Fraction of frames above an IoU threshold                       |
| ViT              | Vision Transformer                | Transformer model applied to image patches                      |

# Appendix

## Appendix A: Suggested ablation table

Table B.1 summarizes ablations that would separate the effects of sequence training,
dynamic cropping, memory tokens, template blurring, and auxiliary memory supervision.

*Table B.1 - Suggested ablations.*

| **Experiment**              | **Memory tokens** | **Dynamic cropping** | **Template blur** | **Memory loss** | **AO** | **SR0.50** | **SR0.75** |
|-----------------------------|-------------------|----------------------|-------------------|-----------------|--------|------------|------------|
| Baseline OSTrack-256 + CE   | No                | No                   | No                | No              | 0.710  | 0.804      | 0.682      |
| Baseline OSTrack-384 + CE   | No                | No                   | No                | No              | 0.737  | 0.832      | 0.708      |
| Sequence-only control       | No                | Yes/No               | No                | No              | TBD    | TBD        | TBD        |
| MemOSTrack without blur     | Yes               | Yes                  | No                | No              | TBD    | TBD        | TBD        |
| MemOSTrack with blur        | Yes               | Yes                  | Yes               | No/Yes          | TBD    | TBD        | TBD        |
| MemOSTrack with memory loss | Yes               | Yes                  | Optional          | Yes             | TBD    | TBD        | TBD        |

## Appendix B: Minimal pseudocode for dynamic training

initialize template from first frame

initialize tracker state from first ground-truth box

initialize memory tokens

for each search frame in the sampled sequence:

crop search region around current tracker state

run MemOSTrack on template, search crop, and memory

compute tracking loss against current ground-truth box

update memory tokens

convert predicted box to image coordinates

set tracker state to predicted box

backpropagate accumulated sequence loss
