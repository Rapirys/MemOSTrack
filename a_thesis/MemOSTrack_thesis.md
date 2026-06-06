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

*Figure 0.1 - High-level idea of the thesis.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>OSTrack baseline -&gt; add memory tokens -&gt; add GRU update</p>
<p>Rewrite sampler for ordered sequences -&gt; use dynamic search cropping</p>
<p>Evaluate MemOSTrack against OSTrack-256 + CE and OSTrack-384 + CE</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

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

*Figure 1.1 - Template-search tracking formulation.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Template crop z from the first frame describes the target appearance.</p>
<p>Search crop x_t from the current frame contains the expected target location.</p>
<p>The tracker predicts the current bounding box inside the search crop.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

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

*Figure 1.2 - Why compact memory is needed.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Full video-token attention grows quadratically with the number of tokens.</p>
<p>A compact memory stream keeps only a small set of recurrent tokens.</p>
<p>The model trades exact past-frame storage for a learned temporal summary.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

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

*Figure 2.1 - Simplified OSTrack baseline architecture.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Template image -&gt; patch embedding -&gt; template tokens</p>
<p>Search image -&gt; patch embedding -&gt; search tokens</p>
<p>Template and search tokens are processed jointly by transformer layers</p>
<p>Prediction head outputs the bounding box</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 2.2 Baseline configuration used in this thesis

The main same-resolution baseline for this thesis is OSTrack-256 + CE. In this name,
256 denotes the 256 x 256 search crop used during tracking, while CE denotes the
candidate elimination mechanism from OSTrack. This baseline is the fairest direct
comparison for MemOSTrack because the proposed model keeps the same search resolution,
the same ViT-B/16 backbone family, and the same CE-enabled token-pruning mechanism, then
adds recurrent memory tokens on top of that configuration. The higher-resolution
OSTrack-384 + CE model is also reported later, but it should be interpreted as a
stronger reference with a larger visual token budget rather than as the same-resolution
baseline.

The baseline input consists of a template crop and a search crop. In the evaluated
configuration, the template crop size is 128 x 128 pixels and the search crop size is
256 x 256 pixels. With the ViT-B/16 patch size, this corresponds to 64 template tokens
and 256 search tokens. Both crops are normalized with the usual ImageNet mean and
standard deviation values commonly used in transformer-based vision models.

The backbone is `vit_base_patch16_224_ce`, initialized from the
`mae_pretrain_vit_base.pth` checkpoint. This detail is important because the quality of
the pretrained backbone has a large influence on tracking performance. Using the same
MAE-pretrained ViT-Base style backbone, the same 256 search resolution, and the same CE
setting makes the experiment directly comparable to the OSTrack-256 + CE configuration
reported in the original OSTrack article.

This description intentionally remains at the level needed for the thesis. Exact
implementation details depend on the codebase, but the important baseline property is
clear: the tracker processes template and search tokens jointly, has no explicit
recurrent memory across frames, and predicts the target from the resulting search
features.

## 2.3 Candidate elimination and token-budget comparison

The original OSTrack framework includes a candidate early elimination mechanism to
improve efficiency by dropping unlikely search tokens \[13\]. In the CE configuration
used here, candidate elimination is applied at three ViT layers with keep ratio
`rho = 0.7`. The implementation keeps template tokens and memory tokens, while pruning
only the search-token stream.

The reported baselines used for final comparison are OSTrack-256 + CE with GOT-10k AO
71.0, SR0.50 80.4, and SR0.75 68.2, and OSTrack-384 + CE with GOT-10k AO 73.7,
SR0.50 83.2, and SR0.75 70.8 \[13\]. In decimal form these values are 0.710, 0.804,
0.682 for OSTrack-256 + CE and 0.737, 0.832, 0.708 for OSTrack-384 + CE. The 256
baseline is the same-resolution comparison, while the 384 baseline is a stronger
high-resolution reference.

The token budget helps interpret the comparison. With a ViT-B/16 backbone, the
128 x 128 template in OSTrack-256 produces 64 template tokens and the 256 x 256 search
region produces 256 search tokens, for 320 visual tokens before candidate elimination.
Adding 64 memory tokens gives MemOSTrack-256 + CE an initial sequence length of 384
tokens. In OSTrack-384, the 192 x 192 template produces 144 template tokens and the
384 x 384 search region produces 576 search tokens, for 720 visual tokens. Therefore,
MemOSTrack-256 + CE has more tokens than OSTrack-256 + CE, but still far fewer
tokens than OSTrack-384 + CE.

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

*Figure 2.2 - Original pair-based sampling.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>A video sequence contains many frames: F1, F2, F3, ..., FT.</p>
<p>Pair-based training samples a template frame and one search frame.</p>
<p>Intermediate frames do not influence the current training sample.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

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

These two limitations motivated the major training-pipeline changes described in Chapter
4. The architecture alone was not enough. A recurrent tracker also needed a recurrent
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

## 3.5 Comparison of the two variants

The two variants represent different stages of the research process. Variant A
asks whether adding memory tokens alone is enough. Variant B asks whether memory
needs an explicit recurrent update. Presenting both is important because the
final architecture was not chosen arbitrarily; it was motivated by the weakness
of the first implementation.

| Component | Variant A: memory tokens only | Variant B: memory + GRU |
|---|---|---|
| Memory tokens | Yes | Yes |
| Explicit recurrent gate | No | Yes |
| Transformer role | Directly updates memory | Produces candidate memory |
| Previous-frame memory | Passed forward implicitly | Used by `GRU1` |
| Previous-layer memory | Used through normal layer flow | Used by `GRU2` |
| Update behavior | Direct transformer output | Gated recurrent update |
| Main risk | Memory can be overwritten | More parameters and harder optimization |
| Purpose | Minimal first prototype | Final controlled memory update |

The no-GRU model is therefore not a failed design to hide. It is part of the
experimental logic of the thesis. It establishes the simplest memory-token
extension and explains why the GRU variant was added.

The two-stage GRU design uses two recurrent sources because memory has to move
in two directions: through time and through network depth. The temporal source
is important because tracking is sequential. If the target gradually changes
appearance, memory from the previous frame may contain a more recent target
representation than the initial template. It also creates a direct differentiable
connection between neighboring frames during backpropagation through time.

The layer-wise source is important because transformer layers refine features
gradually. By using `M_{t,l-1}`, the update allows lower-layer memory
information to influence higher-layer memory without forcing all memory to pass
only through attention over the full token sequence. This creates a shorter path
inside the memory stream. It does not guarantee better performance, but it is a
reasonable architectural hypothesis.

The main notation used in the architecture is summarized below.

| Symbol | Meaning |
|---|---|
| `S_{t,l}` | Search tokens at frame `t` and layer `l` |
| `T_{t,l}` | Template tokens at frame `t` and layer `l` |
| `M_{t,l}` | Final memory tokens at frame `t` and layer `l` |
| `M'_{t,l}` | Candidate memory produced by the transformer layer |
| `M_{t-1,l}` | Memory from the previous frame at the same layer |
| `M_{t,l-1}` | Memory from the previous layer in the current frame |
| `\tilde{M}_{t,l}` | Intermediate memory after the first GRU update |

## 3.6 Forward pass, implementation details, and hyperparameters

For clarity, the final memory-augmented layer can be described as an algorithm.
The layer receives search tokens, template tokens, memory from the previous
layer in the current frame, and memory saved from the previous frame at the same
layer. The tokens are concatenated and processed by the standard transformer
block. The output is split back into search, template, and memory parts. The
memory part is treated as candidate memory and is passed through the two-stage
GRU update.

```python
# Inputs:
#   S: search tokens
#   T: template tokens
#   M_layer: memory from the previous layer in the current frame
#   M_time: memory from the previous frame at the same layer

Y = concat(S, T, M_layer)
Y_out = transformer_layer(Y)
S_out, T_out, M_candidate = split(Y_out)

M_temporal = GRU1(input=M_candidate, hidden=M_time)
M_final = GRU2(input=M_temporal, hidden=M_layer)

return S_out, T_out, M_final
```

This algorithm also shows where implementation errors can occur. Token splitting
must match token concatenation. Memory for each transformer layer must be stored
separately across frames. The recurrent state must be reset at the beginning of a
new video sequence. If memory from one video is carried into another unrelated
video, the recurrent state becomes semantically invalid.

The memory extension also required changes to optimizer parameter grouping. The
project code contained memory-specific parameters such as `backbone.mem_*` and
`backbone.read_mem_embed`. These parameters should not accidentally receive an
unsuitable learning-rate schedule inherited from the pretrained backbone group.

Separate learning rates are important because the model combines pretrained and
newly initialized components. The backbone starts from a pretrained ViT-style
checkpoint and should usually be updated carefully. The memory tokens and
GRU-related parameters start from a less specialized state and may need a larger
or separately controlled learning rate. If the memory parameters learn too
slowly, the baseline template-search path can dominate and memory may be
ignored. If they learn too aggressively, they can disturb the pretrained
representation.

Gradient clipping also becomes more important after introducing recurrent
sequence training. In pair-based training, each sample contributes an independent
loss. In sequence training, losses from several frames are accumulated while
memory is propagated. Dynamic cropping can also create hard examples when early
predictions move the crop away from the target. Clipping the gradient norm is a
practical safeguard against occasional unstable updates.

The architectural comparison with the baseline is summarized below.

| Component | OSTrack-256 + CE | MemOSTrack-256 + CE |
|---|---|---|
| Template tokens | Yes | Yes |
| Search tokens | Yes | Yes |
| Memory tokens | No | Yes |
| Recurrent state | No explicit recurrent state | Memory propagated across frames |
| Backbone type | ViT-style one-stream transformer | ViT-style one-stream transformer with memory |
| Candidate elimination | Used | Used |
| Memory update | Not applicable | Two-stage GRU update |

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

*Figure 4.1 - Sequence-based sampling.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Before: template frame plus one search frame.</p>
<p>After: template frame plus an ordered sequence of search frames.</p>
<p>The memory state is propagated through the sequence.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 4.3 Remaining problem: ground-truth-centered crops

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

## 4.4 Final rewrite: dynamic search cropping

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

*Figure 4.2 - Dynamic search cropping.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Frame t: crop around previous prediction and predict current box.</p>
<p>Frame t+1: crop around the prediction from frame t.</p>
<p>Training becomes closer to the actual inference loop.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*Table 4.1 - Evolution of the sampling pipeline.*

| **Stage**                 | **Input form**       | **Crop center**      | **Memory training quality** | **Inference similarity** |
|---------------------------|----------------------|----------------------|-----------------------------|--------------------------|
| Original OSTrack sampling | Template-search pair | Current ground truth | No recurrent memory         | Low                      |
| First rewrite             | Ordered sequence     | Current ground truth | Memory can be propagated    | Medium                   |
| Final rewrite             | Ordered sequence     | Previous prediction  | Memory sees realistic drift | High                     |

## 4.5 Coordinate transformations in dynamic cropping

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

## 4.6 Backpropagation through time in the training loop

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

## 4.7 Curriculum considerations

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

## 4.8 Why inference-like training is harder

Dynamic cropping makes training harder. With ground-truth-centered crops, the model sees
clean examples even if its previous predictions would have been inaccurate. With dynamic
cropping, an early localization error can move the next crop away from the target
center. The target may appear near the edge of the crop, partially outside the crop, or
surrounded by more distractors. This can reduce training stability.

However, this difficulty is also the point. A tracker deployed in inference does not
receive perfect search crops. If the goal is to train a recurrent memory module, the
memory should be exposed to realistic inputs. Otherwise, the memory may learn to work
only in an idealized setting where every frame is perfectly centered.

This trade-off is important for interpreting the final results. MemOSTrack was
trained with a more demanding pipeline than simple pair-based training. The resulting
score being slightly lower than baseline may partly reflect the difficulty of optimizing
a model under inference-like drift.

## 4.9 Practical consequences for batching

Sequence-based training also affects batching. In pair-based training, each element of a
batch can be processed independently. In sequence training, each element contains
multiple ordered frames and the model must loop over them. This increases memory
consumption and computation time. It also requires careful handling of tensor shapes:
the batch dimension and sequence dimension must not be confused.

The sampler must return not only images, but also the ordered annotations needed to
compute losses and transform predictions between crop coordinates and original image
coordinates. Dynamic cropping adds further complexity because the crop for frame t is
not known until the prediction for frame t-1 has been computed.

This engineering work is a central part of the thesis. Without it, the memory
architecture would not be trained in the way it is intended to be used.

# 5. Auxiliary Memory Supervision

## 5.1 Why memory can be ignored

Adding memory tokens to a strong model does not guarantee that the model will use them.
OSTrack already has a powerful template-search pathway. If the tracker can solve the
training task using template and search tokens alone, the new memory tokens may receive
weak or unimportant gradients. In that case, the architecture contains memory, but the
learned function remains close to the baseline.

This is a common issue when adding auxiliary modules to strong neural networks. The
optimization process often finds the easiest path to reduce the loss. If the template is
clean and the search crop is well-centered, the easiest path may be to rely on the
original OSTrack features. The memory branch then becomes decorative rather than
functional.

The project therefore explored auxiliary memory supervision. The goal was to make memory
tokens directly useful for prediction rather than merely present in the token sequence.

## 5.2 Memory loss heads

A memory loss head is an auxiliary prediction head attached to memory tokens. The main
OSTrack head predicts the target from the ordinary tracking features. The memory head
predicts a target-related output from memory tokens.

*M\_{t,l} -\> Head\_{mem} -\> \hat{B}^{mem}\_t.*

The auxiliary prediction is compared with the ground-truth box:

*L\_{mem} = L\_{box}(\hat{B}^{mem}\_t, B^{gt}\_t).*

The total loss can then be written as:

*L = L\_{main} + λ\_{mem} L\_{mem}.*

Here lambda_mem controls the strength of the auxiliary memory supervision. In principle,
this encourages memory tokens to encode information that is useful for localization. The
memory head does not necessarily have to be used during inference. It can be a
training-time tool that makes the memory state more predictive.

*Figure 5.1 - Auxiliary memory loss head.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Memory tokens are passed to a separate prediction head.</p>
<p>The auxiliary prediction is compared with the ground-truth box.</p>
<p>The loss encourages memory tokens to contain tracking-relevant information.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 5.3 Issues encountered with memory heads

The first issue is that memory tokens are abstract. Search tokens correspond to image
patches, so it is natural to use them for spatial prediction. Memory tokens do not have
the same direct spatial meaning. A memory token may store target identity, appearance,
motion context, or information distributed across several frames. Predicting a box from
such tokens is therefore less straightforward.

The second issue is loss balancing. If the memory loss is too weak, it may not change
the behavior of the model. If it is too strong, it may interfere with the main tracking
objective and damage the pretrained representation. This is especially important because
the baseline is already strong. The auxiliary objective should guide the memory branch
without forcing the whole model to optimize an unstable secondary task.

The third issue is early training noise. At the beginning of training, the memory branch
is not yet reliable. If its predictions are poor, the auxiliary loss can generate noisy
gradients. Since the memory is recurrent, early poor states can influence later frames
in the same sequence. This makes it necessary to tune the loss weight and possibly apply
the memory loss only after a warm-up period.

The fourth issue is shortcut learning through the template. Even with memory loss heads,
the main branch may rely heavily on the clean template. If the template contains a
strong appearance cue, the model may have little incentive to develop a robust memory
representation.

## 5.4 Template blurring as a solution

Template blurring was considered as a way to reduce over-reliance on the initial
template. The idea is not to remove the template completely, but to weaken it enough
that the model benefits from using memory. If the template is too clean, the tracker can
solve many training examples by direct template-search matching. If the template is
partially blurred, the recurrent memory stream becomes more valuable.

*\tilde{z} = Blur(z), \hat{B}\_t = f(\tilde{z}, x_t, M\_{t-1}).*

This acts as a regularization method. It encourages the model to treat the template as
an initial identity cue rather than a perfect appearance reference. The memory can then
store information from later observations. This idea is especially relevant for a
tracker that processes a sequence: the target may be visible more clearly in later
frames than in the first template, or it may change appearance in a way that the first
template cannot represent.

Template blurring must be used carefully. If the blur is too weak, it does not change
the learning problem. If it is too strong, the tracker may lose target identity and
become unstable. The best setting is likely task-dependent and should be selected
through ablation studies. In this thesis, template blurring is presented as an explored
solution to the memory-usage problem, not as a fully optimized technique.

*Figure 5.2 - Template blurring motivation.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Clean template: model can rely mostly on initial appearance.</p>
<p>Blurred template: initial appearance is weakened but not removed.</p>
<p>Memory stream is encouraged to carry recent target information.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*Table 5.1 - Memory supervision issues and responses.*

| **Issue**                   | **Cause**                                    | **Response explored in this thesis**      |
|-----------------------------|----------------------------------------------|-------------------------------------------|
| Memory may be ignored       | Baseline template-search path is strong      | Auxiliary memory loss heads               |
| Template dominates          | Clean initial template is highly informative | Template blurring during training         |
| Memory head instability     | Memory tokens are not naturally spatial      | Careful loss weighting and analysis       |
| Recurrent error propagation | Bad early memory affects later frames        | Inference-like training exposes the issue |

## 5.5 What memory supervision should prove

The expected behavior is qualitative: the model should use the template to initialize
identity, search tokens to observe the current frame, and memory tokens to preserve and
update target information over time. If the memory branch is useful, blurring the
template should not completely destroy tracking performance because the tracker can
compensate with temporal information.

This idea also makes the thesis more realistic. Adding memory is not only about
architecture; it is about creating a training environment in which memory is needed.
Memory loss heads, template blurring, and dynamic cropping are all attempts to prevent
the model from solving the task through the simplest baseline pathway.

# 6. Experimental Setup

## 6.1 Dataset and protocol

The experiments are done on the GOT-10K dataset. GOT-10k is a
generic object tracking benchmark with more than 10,000 video segments and more than 1.5
million labeled bounding boxes \[20\]. The testing data is not labled and are evaluated by the offcial dataset website.
The baseline OsTrack paper publishes a detailed report of their tracker perfomance on Got-10k in different configurations. 
The model is trained exclusively on the GOT-10k data without external samples from different datasets. 

The final comparison uses AO, SR0.50, and SR0.75. The baseline values are taken from
reported CE-enabled OSTrack results: OSTrack-256 + CE with AO 0.710, SR0.50 0.804,
and SR0.75 0.682, and OSTrack-384 + CE with AO 0.737, SR0.50 0.832, and SR0.75
0.708 \[13\]. The MemOSTrack values are the experimental results obtained for this
thesis.


## 6.2 Model configuration

For the testing purposes the baseline with 256 pixels search crop with CE pruning was chosen.

The model configuration can be summarized as follows: template crop size 128 x 128
pixels, search crop size 256 x 256 pixels, 64 template tokens, 256 search tokens, 64
memory tokens, ViT-style OSTrack backbone, memory tokens inside the transformer token
sequence, two-stage GRU recurrent update, candidate elimination enabled with keep ratio
0.7, causal consecutive sequence sampler, and dynamic search cropping based on previous
predictions in the final pipeline. The recurrent training was initially performed with
20-frame search rollouts; in the later stages, this was reduced to 15-frame rollouts.

This configuration is an experimental prototype. It is not claimed to be the optimal
memory-augmented tracker. The purpose is to investigate whether the idea is feasible and
how it behaves compared with a strong baseline.

*Table 6.1 - Main model and training configuration.*


## 6.3 Training details

The training process required several implementation changes beyond normal
hyperparameter selection. First, the sampler had to return ordered frame sequences.
Second, the training loop had to propagate memory across frames in each sequence. Third,
dynamic cropping required the previous predicted box to be converted into the coordinate
system of the next image. Fourth, new memory parameters had to be included correctly in
the optimizer.

The experiment reported in this thesis was evaluated after the available training run.
The user-provided development note used the phrase '58 training iterations'. In the
final submitted thesis, this phrase should be checked against the actual training logs.
If the value means epochs, the thesis should say '58 epochs'. If it means checkpoints,
it should say '58 checkpoints'. If it literally means optimizer iterations, then the
model was severely undertrained and the result should be presented as preliminary. This
draft uses the cautious phrase 'available training run' unless the exact meaning is
confirmed.

This caution is important for scientific accuracy. A thesis can report negative or
inconclusive results, but the training budget must be described precisely. The result
should not be overstated if the training run was shorter than the baseline training
schedule.

## 6.4 Hyperparameter and optimizer considerations

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

## 6.5 Evaluation protocol

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

## 6.6 Reproducibility considerations

For reproducibility, the final implementation should record the exact code revision,
dataset split, training command, random seed if used, hardware, number of epochs or
optimizer steps, checkpoint used for evaluation, and configuration file. These details
are especially important for memory-based trackers because small changes in sequence
sampling can produce different training behavior.

The thesis should also clearly separate external baseline numbers from the author's own
results. The OSTrack baseline numbers are reported values from the literature \[13\].
The MemOSTrack numbers are experimental results from this work. Presenting them in
the same table is useful, but their sources should be distinct.

# 7. Results and Discussion

## 7.1 Quantitative results

The final measured result for MemOSTrack-256 + CE is shown in Table 7.1. The model
achieved AO 0.729, SR0.50 0.823, and SR0.75 0.693. The reported OSTrack-256 + CE
baseline achieves AO 0.710, SR0.50 0.804, and SR0.75 0.682. The reported OSTrack-384 +
CE baseline achieves AO 0.737, SR0.50 0.832, and SR0.75 0.708 \[13\].

*Table 7.1 - Final quantitative comparison.*

| **Method**               | **AO** | **SR0.50** | **SR0.75** | **MemOSTrack - method AO** | **MemOSTrack - method SR0.50** | **MemOSTrack - method SR0.75** |
|--------------------------|-------:|-----------:|-----------:|---------------------------:|-------------------------------:|-------------------------------:|
| MemOSTrack-256 + CE   | 0.729  | 0.823      | 0.693      | -                          | -                              | -                              |
| OSTrack-256 + CE         | 0.710  | 0.804      | 0.682      | +0.019                     | +0.019                         | +0.011                         |
| OSTrack-384 + CE         | 0.737  | 0.832      | 0.708      | -0.008                     | -0.009                         | -0.015                         |

*Figure 7.1 - Metric comparison placeholder.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Bar chart placeholder: MemOSTrack-256 + CE vs OSTrack-256 + CE vs OSTrack-384 + CE.</p>
<p>Metrics: AO, SR0.50, SR0.75.</p>
<p>Observed gaps: +0.019/+0.019/+0.011 against OSTrack-256 + CE and
-0.008/-0.009/-0.015 against OSTrack-384 + CE.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

The result improves over the same-resolution OSTrack-256 + CE baseline on all three
metrics. The gains are 0.019 AO, 0.019 SR0.50, and 0.011 SR0.75. This indicates that,
in the 256 CE setting, the memory-augmented model is not merely adding complexity; it
achieves a measurable accuracy improvement over the corresponding OSTrack baseline.

However, the result remains below OSTrack-384 + CE. The gaps are 0.008 AO, 0.009
SR0.50, and 0.015 SR0.75. This means that the current memory-augmented 256 model is
competitive with the stronger high-resolution baseline, but it does not surpass it.
The final interpretation is therefore mixed: recurrent memory improves the 256 CE
setting, but increasing the search resolution to 384 still gives the best accuracy among
the compared models.

## 7.2 Token-budget interpretation

The result should be interpreted together with the number of tokens processed by the ViT
backbone. With a patch size of 16, the OSTrack-256 setting uses a 128 x 128 template and
a 256 x 256 search region. This gives 64 template tokens and 256 search tokens, for 320
visual tokens before candidate elimination. MemOSTrack-256 + CE adds 64 memory
tokens, increasing the initial sequence length to 384 tokens.

The OSTrack-384 setting uses a 192 x 192 template and a 384 x 384 search region. This
gives 144 template tokens and 576 search tokens, for 720 visual tokens before candidate
elimination. If the same 64 memory tokens were added to a 384 setting, the initial
sequence length would be 784 tokens.

*Table 7.2 - Input token counts before candidate elimination.*

| **Model setting** | **Template tokens** | **Search tokens** | **Memory tokens** | **Initial tokens** |
|-------------------|--------------------:|------------------:|------------------:|-------------------:|
| OSTrack-256 + CE | 64 | 256 | 0 | 320 |
| MemOSTrack-256 + CE | 64 | 256 | 64 | 384 |
| OSTrack-384 + CE | 144 | 576 | 0 | 720 |
| Hypothetical MemOSTrack-384 + CE | 144 | 576 | 64 | 784 |

Candidate elimination changes the active sequence length inside the backbone. In the
configuration used here, search tokens are pruned at three layers with keep ratio
`rho = 0.7`, while template and memory tokens are kept. Using the implementation's
`ceil(0.7 * current_search_tokens)` rule, the 256 search stream is reduced from 256
tokens to 180, then 126, then 89. The 384 search stream is reduced from 576 tokens to
404, then 283, then 199.

*Table 7.3 - Active token counts after CE pruning.*

| **Model setting** | **Initial tokens** | **After CE 1** | **After CE 2** | **After CE 3** |
|-------------------|-------------------:|---------------:|---------------:|---------------:|
| OSTrack-256 + CE | 320 | 244 | 190 | 153 |
| MemOSTrack-256 + CE | 384 | 308 | 254 | 217 |
| OSTrack-384 + CE | 720 | 548 | 427 | 343 |
| Hypothetical MemOSTrack-384 + CE | 784 | 612 | 491 | 407 |

This token-budget view explains why the two comparisons should be discussed
separately. MemOSTrack-256 + CE has a 20 percent larger initial token sequence than
OSTrack-256 + CE because of the 64 memory tokens. At the same time, it is still much
smaller than OSTrack-384 + CE: 384 initial tokens versus 720, and 217 active tokens
versus 343 after the third CE pruning point. The remaining gap to OSTrack-384 + CE may
therefore be partly explained by the larger visual token budget of the 384 model.

## 7.3 What worked technically

In addition to the quantitative improvement over OSTrack-256 + CE, several technical
goals were achieved. The architecture was modified to include memory tokens. The memory
tokens were updated through a two-stage GRU mechanism. The sampler was rewritten to
produce ordered video sequences. The training pipeline was rewritten to use
inference-like dynamic cropping. Auxiliary memory supervision and template blurring
were studied as responses to memory underuse.

These are meaningful engineering and research contributions. A bachelor thesis is not
required to produce a new state-of-the-art tracker. It is required to formulate a
problem, implement a solution, evaluate it, and analyze the outcome. This work satisfies
that structure.

## 7.4 Limitations

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

The fourth limitation is uncertainty in the exact training budget description. The final
thesis must verify whether the available run corresponds to epochs, checkpoints, or
optimizer iterations. This affects how strongly the result should be interpreted.

## 7.5 Qualitative analysis that should accompany the final version

In addition to the metric table, a final thesis submission should include qualitative
examples. These examples do not need to prove that the proposed method is better; their
purpose is to explain model behavior. Useful examples would include one sequence where
MemOSTrack follows the target correctly, one sequence where it drifts, and one sequence
where both OSTrack and MemOSTrack fail. For each example, the figure should show several
frames with predicted and ground-truth boxes.

The qualitative analysis should focus on memory-related questions. Does the tracker
recover after partial occlusion? Does it follow the target when the appearance changes?
Does it drift to a distractor after a wrong crop? Does template blurring make the
tracker less stable in the first frames? These questions are more informative than
simply showing successful frames.

*Figure 7.2 - Qualitative example placeholder.*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p>■</p>
<p>Replace this placeholder with frames from one representative sequence.</p>
<p>Show ground truth and predicted boxes over time.</p>
<p>Use the caption to explain whether memory helped, failed, or behaved
neutrally.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 7.6 Interpretation of the mixed result

The final result is neither a simple failure nor a general victory over OSTrack. It is
positive relative to the same-resolution OSTrack-256 + CE baseline, where MemOSTrack is
higher on all three GOT-10k metrics. At the same time, it is not enough to surpass the
OSTrack-384 + CE baseline, which benefits from a much larger search-region token budget.

The correct conclusion is therefore specific: in the evaluated configuration, adding 64
memory tokens and a GRU-based recurrent update improves the 256 CE setting, but it does
not replace the accuracy benefit of the 384 search resolution. This distinction is
important. It supports the memory hypothesis at the same resolution while avoiding the
overstated claim that MemOSTrack is generally better than OSTrack.

This also affects the writing style of the thesis. The thesis should avoid broad claims
such as 'the proposed method outperforms OSTrack'. It should instead use precise phrases
such as 'outperforms OSTrack-256 + CE', 'remains below OSTrack-384 + CE', and
'investigates the effect of recurrent memory tokens'. That framing is scientifically
accurate and appropriate for a bachelor thesis.

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
and different values of lambda_mem. Template blurring should also be varied by blur
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
of AO 0.737, SR0.50 0.832, and SR0.75 0.708. Therefore, the proposed method improves
the same-resolution 256 CE setting, but it does not outperform the higher-resolution
384 CE model. The result provides a useful foundation for future work on memory
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
Relation Modeling for Tracking: A One-Stream Framework,' ECCV, 2022. The paper reports
OSTrack-256 + CE with GOT-10k AO 71.0, SR0.50 80.4, and SR0.75 68.2, and OSTrack-384 +
CE with GOT-10k AO 73.7, SR0.50 83.2, and SR0.75 70.8.

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
baseline, but lower than the reported OSTrack-384 + CE baseline. The main conclusion is
that recurrent memory improves the same-resolution 256 CE setting, but does not yet
surpass the higher-resolution 384 CE model. Further optimization and ablation studies
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

The following table should be filled if additional experiments are performed.

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

## Appendix C: Notes for final thesis completion

Before final submission, the placeholder diagrams should be replaced by clean vector
figures. The most important diagrams are the OSTrack baseline architecture, the
MemOSTrack architecture, the two-stage GRU memory update, the sequence sampler, and
dynamic search cropping. The phrase describing the training budget must also be checked
against the training logs. If the run contains 58 epochs, the thesis should state 58
epochs. If it contains 58 optimizer iterations, the result should be labeled
preliminary.
