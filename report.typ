#import "@preview/manuscr-ismin:0.3.1": *

#show: manuscr-ismin.with(
  lang: "en",
  title: [Research Project Report],
  subtitle: [Literature Review & Experiments for Graph-based Reasoning and Efficient Quasi‑Indoor–Outdoor Robot Navigation],
  school: (
    name: [ENSTA],
    subname: [Campus de Paris-Saclay],
  ),
  course: (
    ue: "",
    ecue: "",
    name: [Guided Research Project],
    subname: [],
  ),
  authors: (
    (
      name: [Duc Nguyen],
      affiliation: "Telecom Paris",
      email: "dnguyen-25@ip-paris.fr",
    ),
  ),
  mentor1: (
    role: "Supervisor",
    name: [Dr. #smallcaps[Zhi Yan]],
    email: "zhi.yan@ensta.fr",
  ),
  mentor2: (
    role: "",
    name: [],
    email: "pchhch@emse.fr",
  ),
  academic-year: [2025-2026],
  header: [#h(1fr) #upper[Guided Research Project]],
  logo: image("assets/logo_ensta_2025.jpg", width: 7.3cm),
  date: "04/05/26",
  latex-look: false,
)

/*
 * Tables
 * Retirez celles dont vous ne vous servez pas
 */

// #outline(indent: auto)

// #heading(numbering: none)[Table des figures] <fig_outline>
// #outline(target: figure.where(kind: image), title: none)

// #heading(numbering: none)[Table des tableaux]
// #outline(target: figure.where(kind: table), title: none)

// #heading(numbering: none)[Table des équations]
// #outline(target: figure.where(kind: "equation"), title: none)

// #heading(numbering: none)[Table des listages]
// #outline(target: figure.where(kind: raw), title: none)

// #pagebreak()

/* Reste du document */

#set par(first-line-indent: (
  amount: 1em,
  all: true,
))

= Introduction

University campuses, hospital complexes, and corporate parks form a distinct class of outdoor environments. They are large in scale, semantically structured, and populated by humans who expect natural interaction with autonomous systems. In such settings, robots are required to perform tasks that go beyond local navigation. A delivery robot may need to transport supplies across multiple buildings using only a destination name. A tour guide robot must interpret queries such as "take me to the admissions office" and produce a coherent route. A hospital porter robot must operate reliably despite dynamic changes such as temporary closures or shifting pedestrian flow.

Despite their diversity, these tasks share a common structure: the robot operates within a *bounded site*—large enough to require semantic reasoning about destinations, yet coherent enough to exhibit consistent organizational patterns.

This coherence is not accidental. Campuses and similar environments are designed according to established planning principles. Buildings are grouped into functional zones, pathways reflect natural circulation, entrances align with expected flows, and open spaces mediate transitions between areas. The resulting layout encodes a hierarchical semantic organization that is legible to humans. In effect, the environment itself expresses a structured "argument" about how it should be used.

#figure(
  image("images/intro_ipparis.jpg", width: 90%),
  caption: [IP Paris Campus Organization]
)

We argue that this structure constitutes *a largely unexploited inductive bias* for robot navigation. Existing approaches to hierarchical scene representation are predominantly bottom-up: they derive structure from geometric and visual cues without incorporating prior knowledge of functional organization. As a result, the learned hierarchy lacks grounding in the design intent already embedded in the environment. Incorporating this prior offers several advantages: it enables initialization of a meaningful structural scaffold before perception, constrains and guides entity discovery, and supports reasoning about unexplored regions by analogy to known ones.

This report presents two contributions. First, we provide a literature survey spanning 3D scene graphs, construction pipelines, outdoor semantic mapping, scene graph-based navigation, and robot learning paradigms. This survey situates our work within the current state of the art and highlights the gaps that motivate our approach. Second, we introduce a site-scale semantic navigation framework designed around four principles: memory efficiency, planning efficiency at inference time, flexibility to open-vocabulary queries and unseen environments, and accessibility through natural language interaction.

The proposed framework consists of two components with a clear separation of responsibilities:
#pagebreak()
#columns(2, gutter: 8pt)[
   The *global planning component* constructs and queries a hierarchical 3D scene graph aligned with the site’s planning structure. Each node is described by a free-text caption and a dense language embedding, avoiding fixed taxonomies. The graph is initialized using OpenStreetMap data to create a structural backbone of buildings and pathways, and is incrementally refined with finer-grained entities as the robot explores. Natural language goals are processed by a language model that decomposes queries into semantic subgoals, which are resolved to graph nodes via hierarchical embedding search. These nodes are then mapped onto a navigation graph—constructed from pathway networks and augmented with waypoints for open spaces—over which shortest-path planning produces a sequence of waypoints.

   The *local motion component* operates independently of the global scene graph. Given the waypoint sequence, it generates executable trajectories between consecutive waypoints using the robot’s current perception. Visual foundation models, including SAM2 and VGGT, are used to extract features encoding both semantic instances and local 3D structure. This separation ensures that global reasoning and local control operate at appropriate scales, with the waypoint sequence serving as the only interface between them.
  #colbreak()

  #align(center)[
    #figure(
      image("images/intro_osg.png", width: 90%),
      caption: [Global Planning]
    )
    #figure(
      image("images/intro_local_motion.png", width: 90%),
      caption: [Local Motion]
    )
  ]
]

= Background: Scene Graph Generation

#cite(<cite6>, form: "prose") survey 3D scene representations for robotics across five functional modules, providing a useful taxonomy for motivating the representational choices in this work. Geometric methods — voxel grids, occupancy maps, point clouds — dominate deployed navigation systems for their real-time efficiency and metric accuracy, but carry no semantic content and cannot respond to language-specified goals. Neural representations such as NeRF embed language features directly into the scene via CLIP integration, enabling open-vocabulary spatial queries, but their rendering cost precludes real-time closed-loop control. 3D Gaussian Splatting recovers real-time performance through explicit primitive rendering, yet remains optimized for photometric reconstruction rather than semantic organization. Foundation model approaches — NLMap, Scene-LLM — provide language grounding and open-vocabulary querying but struggle with metric accuracy at scale and introduce inference latency incompatible with reactive local control.

The survey's conclusion is that no single representation is optimal across all navigation scenarios. For a campus navigation robot, the requirements decompose naturally along the same lines: global goal specification and route planning require semantic hierarchy and language grounding, while local trajectory execution requires real-time geometric precision. Scene graphs, which the survey identifies as the natural interface between language goals and metric planning, address the former — providing structured semantic organization at multiple levels of abstraction. Visual foundation models address the latter — providing geometry-aware perceptual features at the frequency local motion demands. The proposed framework is grounded in this decomposition.

#figure(
    image("images/review_scene_repr_compare.png"),
    caption: [Strength and weakness of various scene representations across several dimensions @cite6]
)
  
== Scene Graph

#figure(
  image("images/review_SG.jpeg", width: 80%),
  caption: [Generalized Hierarchical Scene Graph Schema]
)

A scene graph is a directed graph data structure that represents objects and their relationships within a scene. Formally, a scene graph is defined as a tuple $G=(O,R,E)$.

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Set*], [*Element Format*], [*Meaning*],
  ),
  $O$, [
    $o_i$: the object itself\
    $c_i$: semantic category\
    $a_i$: additional properties
  ], [the set of object instances in the scene],
  $R$, [
    $r_(i arrow j)$: relationship between objects $o_i$ and $o_j$.
  ],
  [relationships between objects],
  $E$, [
    $(o_i, r_(i arrow j), o_j)$: object $o_i$ is related to object $o_j$ via relation $r_(i arrow j)$
  ], [
    the edge set of the scene graph
  ]
)

A *3D scene graph* extends this formulation by introducing *hierarchical structure*.

It is defined as a hierarchical multi-graph $G = (V, E)$ where:
- the vertex set $V$ is partitioned into $K$ levels: $V = union_(k=1)^K V_k, V_i inter V_j = emptyset$
- Each subset $V_k$ represents entities at a specific level of abstraction (e.g rooms, buildings)
- For any vertex $v in V_k$: edges may only connect to vertices in adjacent or identical levels:

#align(center)[
  $v arrow u "iff" u in V_(k-1) union V_k union V_(k+1)$
]
This ensures that relationships remain locally consistent within the hierarchy.

== History of Scene Graph Generation

Scene graph generation has evolved through three broad phases:
- 2D visual relationship detection,
- 3D spatial scene graphs, and
- open-vocabulary hierarchical representations enabled by foundation models.

=== 2D Scene Graph Generation

#columns(2, gutter: 8pt)[
  Early work formulates scene graph generation as visual relationship detection: object instances are detected and pairwise relationships are classified from images @cite_lu2016. The release of Visual Genome @cite_krishna2017visual established a large-scale benchmark and standardized the task as supervised prediction over fixed object and predicate vocabularies.
  
  Subsequent methods introduced contextual reasoning. Message-passing networks @cite_xu2017 and later attention-based models @cite_tang2019 enabled joint inference over objects and relationships, improving global consistency. Transformer-based approaches such as RelTR @cite_cong2023 further unified the pipeline by predicting triplets end-to-end.
  
  However, 2D methods remain limited by viewpoint-dependent geometry and closed vocabularies, restricting their applicability to real-world navigation.

  #colbreak()

  #figure(
    image("images/review_SG_evolution.png"),
    caption: [Scene Graph Evolution]
  )
]

=== 3D Scene Graph Generation

3D scene graphs address the lack of spatial grounding by operating directly in metric space. #cite(<cite_armeni2019>, form: "prose") introduced a hierarchical formulation spanning buildings, rooms, and objects, establishing the structural template for subsequent work.

Learning-based methods such as ESCO @cite_wald2020 construct graphs from RGB-D data using geometric and visual features, while systems like HYDRA @cite_hughes2022 demonstrate real-time, incremental construction on mobile robots.

Despite these advances, most 3D approaches remain closed-vocabulary and rely on bottom-up structure inferred solely from sensor data.

=== Open-Vocabulary and Foundation Model Era

Recent work leverages vision-language models to overcome vocabulary limitations. ConceptGraphs @cite_gu2024 associates CLIP embeddings @cite_clip with 3D entities, enabling open-vocabulary querying without retraining.

HOV-SG @cite_werby2024 extends this paradigm to hierarchical representations, supporting natural language queries over building-scale environments. It demonstrates that open-vocabulary hierarchical scene graphs can be constructed incrementally from sensor data.

However, these methods still rely on bottom-up construction and do not incorporate structural priors inherent to designed environments.

= Methodology

During our investigation, we identified a closely related framework, CausalNav @cite_duan2026, whose data structure and pipeline align largely with our objectives. Our approach builds upon this foundation while introducing modifications tailored to site-scale semantic navigation.

== Scene Graph Representation

#columns(2, gutter: 8pt)[
  CausalNav @cite_duan2026 fuses offline map data with real-time open-vocabulary perception into a multi-level hierarchical structure. The Embodied Graph serves simultaneously as a *scene operator* — encoding spatial and semantic relationships among environment entities — and a *memory tank* — storing the robot's historical trajectory as navigable context.

  Formally, the *Embodied Graph* is defined as $G = (V, E)$ where $V$ is partitioned into 5 levels:

  $
  V = V_S union V_Z union V_A union V_O union V_"ego"
  $
  
  Each level encodes a distinct granularity:

  #colbreak()
  #figure(
    image("images/method_embodied_graph.png"),
    caption: [Embodied Graph]
  )
]

- *Site node* $V_S$: single root node representing the entire environment.
- *Zone nodes* $V_Z$: represent coarse site-level entities (buildings, zones, landmarks) initialized from offline map data (OpenStreetMap or equivalent). Each node carries a free-text description and a dense language embedding $bold(e)_Z in RR^d$ derived from a vision-language encoder.
- *Functional area nodes* $V_A$: represent group of spatially co-located and semantically consistent objects. They are constructed via geometric clustering (e.g., DBSCAN), with embedding-based coherence filtering. A LLM is used only to assign a semantic label after the area stabilizes.
- *Object nodes* $V_O$: represent fine-grained instances detected during exploration. Each object node $v_O$ is associated with a bounding volume, a natural language label, and an embedding $bold(e)_O$.
- *Ego-vehicle nodes* $V_"ego"$ encode the robot's historical trajectory, with each node representing a visited pose stamped with the corresponding local map state. These form a temporal chain that supports path reuse during global planning.

The edge set $E = E_"spatial" union E_"hier" union E_"traj"$ contains three edge types:

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Set*], [*Edge Type*], [*Meaning*],
  ),
  $E_"spatial"$, [*Spatial edges*], [geometric relationships between object-level nodes (proximity, containment, adjacency, ...)],
  $E_"hier"$, [*Hierarchical edges*],
  [
    parent-child containment between adjacent levels.
  ],
  $E_"traj"$, [*Trajectory edges*], [
    chain consecutive ego-vehicle nodes into a navigable history graph.
  ]
)

== Scene Graph Construction Pipeline

The scene graph is constructed incrementally through a combination of *top-down initialization* and *bottom-up perception*, with intermediate abstraction layers synthesized during exploration. The process consists of four components that operate concurrently during deployment.

#columns(2, gutter: 8pt)[
  *Offline Initialization*: Prior to exploration, site and zone nodes are extracted from OpenStreetMap data:
  - A single site node $v_S$ represents the global environment
  - Zone nodes $v_Z$ represent buildings and outdoor regions (e.g plazas, parking areas, ...).
  
  Each node is initialized with geometric information (e.g., polygon or centroid), a metadata-derived textual description, and a precomputed language embedding. Pathway geometries are stored separately to support later navigation graph construction. This stage ensures that the robot begins operation with a partially structured representation rather than an empty graph.
  
  #colbreak()

  #figure(
    image("images/method_offline_init.png", width: 90%),
    caption: [Offline Initialization]
  )
]

#figure(
  image("images/method_online_refine.png"),
  caption: [Online Refinement]
)

*Online Refinement*: As the robot explores the environment, object-level nodes are incrementally constructed from sensor observations.

At each timestep t, instance masks are extracted from RGB input using SAM2 in automatic mode, without relying on predefined category labels. For each detected instance $i$, an image crop is encoded using CLIP to obtain a language-aligned embedding $bold(e)_i in RR^d$. The corresponding 3D position $bold(p)_i in RR^3$ is recovered by projecting the mask onto the synchronized LiDAR point cloud and computing the centroid of the resulting cluster.

Each new detected instance is used to augment the *Embodied Graph* through a two-stage process:
- *Spatial Gating*: find existing nodes closed to new instance
- *Semantic matching*: compute CLIP embedding cosine similarity with spatially close nodes

Formally, let $cal(N)$ denote the set of current object nodes.

A spatial gating step first restricts candidate matches to:
$
cal(N)_r = \{ j in cal(N) | \|bold(p)_i - bold(p)_j\| < r \}
$
Among these candidates, semantic similarity is evaluated:
$
"ssim"_(i j) = cos(bold(e)_i, bold(e)_j)
$

- If $max_(j in cal(N)_r) "ssim"_(i j) > tau$, the node $j^*$ is updated using EMA over embedding and geometry.
- Otherwise, a new node is inserted.

This matching procedure enforces spatial consistency and maintains temporal semantic robustness.

Functional area nodes are constructed as an intermediate abstraction by grouping object nodes into coherent regions, which is driven primarily by geometry and topology.

Let $cal(O)_k subset V_O$ denote a candidate cluster.
- In indoor environments, floor boundaries are inferred from the vertical distribution of observations, and rooms are obtained via graph partitioning over the navigable topology, with doorways acting as separators.
- In outdoor environments, object nodes are grouped using density-based clustering (e.g., DBSCAN) over their 3D positions. For semantic consistency, we apply a filter based on embedding variance:$
"Var"(bold(e)_i | o_i in cal(O)_k) < epsilon
$
Clusters that satisfy this condition are promoted to functional area nodes. Once a region stabilizes, a LLM is invoked to generate a text description and compute its embedding. In this formulation, geometric and topological cues determine the structure of functional areas, while language is used only to assign semantic meaning.

*Dynamic Object Filtering* is applied to maintain a stable and persistent representation. Each detected instance is tracked over a temporal window of $k$ steps, forming a spatial-temporal trajectory corridor. Objects whose displacement exceeds a threshold $delta$ are considered dynamic and discarded. This approach is more robust than instantaneous velocity filtering, as it handles intermittently moving objects such as vehicles that pause and resume motion.

== Global Planning

Given a natural language goal $q$, the planning process proceeds in two stages:
- decomposition into a sequence of hierarchically grounded keyframes, and
- resolution of each keyframe to a node in the scene graph.

#figure(
  image("images/method_query_decompose.png"),
  caption: [Natural Language Query Decomposition]
)

*Query Decomposition and Hierarchical Resolution*:

A direct mapping from $q$ to a set of object or place labels is insufficient for reliable retrieval. For example, decomposing "make a coffee and bring it to the meeting room" into "kitchen", "meeting room" ignores spatial context, leading to ambiguity when multiple instances exist across the environment. Instead, the query is decomposed into a sequence of K keyframes, each represented as a tuple of descriptions aligned with the graph hierarchy:
$
\{(d_k^Z, d_k^A, d_k^O)\}_{k=1}^K,
$
where $d_k^Z, d_k^A, and d_k^O$ correspond to zone-, area-, and object-level descriptions, respectively.

The decomposition is generated by a LLM conditioned on the query $q$, the robot's current context (e.g., its associated zone and area). This enables context-aware disambiguation; for instance, references such as "the meeting room" are grounded relative to the robot’s current environment rather than matched globally.

Each keyframe is then resolved through hierarchical retrieval over the graph. Resolution proceeds top-down, first selecting a zone, then an area within that zone, and finally an object within the area:

$
v_Z^* &= "argmax"_(v in V_Z) &&cos(bold(e)_v, "CLIP"(d_k^Z)),\
v_A^* &= "argmax"_(v in V_A, "p"(v)=v_Z^*) &&cos(bold(e)_v, "CLIP"(d_k^A)),\
v_O^* &= "argmax"_(v in V_O, "p"(v)=v_A^*) &&cos(bold(e)_v, "CLIP"(d_k^O)).
$

The hierarchical constraint ensures spatial coherence by restricting search at each level to the sub-tree rooted at the previously selected node. This prevents mismatches across distant regions (e.g., selecting a kitchen in one building and a meeting room in another). Notably, the language model is used only once for decomposition; all subsequent retrieval operations rely on embedding similarity, ensuring efficient inference.

When the robot’s metric position $bold(p)_"ego"$ is available, candidate nodes can be re-ranked using a hybrid score that balances semantic relevance and spatial proximity:
$
eta(v) = beta dot "spatial"(v, bold(p)_"ego") + (1 - beta) dot cos(bold(e)_v, "CLIP"(d_k)).
$

*Navigation Graph and Path Planning*

#figure(
  image("images/method_nav_plan.png"),
  caption: [Navigation Graph & Path Planning Pipeline]
)

Resolving the keyframes sequence yields a corresponding sequence of target nodes $(v_1^*, dots, v_K^*)$. These nodes must be translated into an executable trajectory in metric space. To this end, planning is performed over a separate navigation graph $G_"nav" = (W, P)$, where $W$ denotes waypoints and $P$ denotes traversable paths.

The navigation graph is initialized from OSM pathway geometries, providing a structured backbone of walkable routes across the environment. However, such data is often incomplete in open spaces (e.g., plazas or courtyards). To ensure full coverage, additional waypoints are generated via Voronoi decomposition over obstacle-free regions of the local map. This results in a dense, navigable representation that complements the structured pathways.

Each resolved target node $v_k^*$ is projected onto its nearest waypoint in $W$, anchoring abstract semantic goals to concrete spatial locations. A global path is then computed by applying Dijkstra’s algorithm over $G_"nav"$ between consecutive waypoints.

To improve efficiency and robustness, edge costs are adapted based on experience. If the robot has previously traversed a region, the corresponding edges are assigned lower cost, encouraging reuse of known safe paths. In unexplored regions, edge weights default to Euclidean distance derived from the underlying geometry. The resulting sequence of waypoints forms the global plan, which is passed to the local motion module for execution.

*Coarse waypoint planning.*

A waypoint sequence is produced by Dijkstra's algorithm over $cal(G)_"nav"$ between consecutive projected targets. Waypoints are intentionally kept coarse — spaced at Voronoi-determined intervals rather than densified into a full trajectory — because fine-grained trajectory generation is delegated to the local motion component. The waypoint sequence serves as a goal-conditioning signal: it tells the local policy *where* to go at each step, while leaving *how* to get there — speed regulation, pedestrian avoidance, yielding behavior — to the learned policy. Collapsing global planning and local execution into a single dense trajectory would remove this flexibility and force social compliance to be hand-coded rather than learned.

== Local Planner

The local planner module converts coarse waypoint plans from the global planner into executable robot trajectories. Rather than performing explicit geometric trajectory optimization, we formulate local navigation as a conditional generative modeling problem: given the current scene observation and a short horizon of future waypoints, the policy generates socially compliant motion trajectories in a receding-horizon manner.

We adopt a Diffusion Policy framework @cite_chi2025 trained on demonstrations collected in populated campus environments. Social behaviors—such as yielding, speed modulation in crowds, and collision avoidance—are not explicitly programmed, but emerge from imitation of human demonstrations in dense multi-agent settings.

#figure(
  image("images/method_local_motion.png", width: 90%),
  caption: [Waypoint-conditioned Diffusion Policy with implicit social compliance]
)

=== Perceptual Feature Extraction

The local motion policy operates on latent visual representations extracted from complementary semantic and geometric foundation models:

*SAM2 semantic features*:
Given the current RGB observation, SAM2 @cite_ravi2024 processes the image through its Hiera encoder to produce a dense latent representation $bold(z)_t^"sem" in RR^(d_s)$. Rather than using only the final segmentation outputs, we retain intermediate encoder features, which capture higher-level semantic structure including object appearance, spatial layout, and contextual relationships.

*VGGT geometric features*:
In parallel, VGGT @cite_wang2025 extracts a geometric latent representation $bold(z)_t^"geo" in RR^{d_g} $, encoding local 3D scene geometry directly from RGB observations. This representation captures metric structure such as obstacle proximity, surface layout, and relative depth relationships without requiring explicit depth sensors or online reconstruction.

*Latent fusion*: The semantic and geometric features are fused into a joint scene representation:

$
bold(z)_t = "Fuse"(bold(z)^"sem", bold(z)^"geo")
$

where $"Fuse"(dot)$ denotes feature concatenation followed by a learned projection network.

The resulting latent representation provides a compact encoding of both scene semantics and local geometry, which conditions the downstream diffusion policy.

=== Diffusion Policy

*Formulation*: The local policy models a conditional distribution over short-horizon trajectories given the current observation and goal waypoint. At time $t$, the observation is:
$
bold(o)_t = {bold(z)_t, bold(s)_t, bold(W)_t}
$

where:
- $bold(z)_t in RR^(d_s+d_g)$ is the fused scene latent,
- $bold(s)_t in RR^d$ is the robot state (pose and velocity), and
- $bold(W)_t = {bold(w)_(t+1), dots, bold(w)_(t+K)}$: K future waypoints

The policy outputs a horizon of 2D velocity increments in the robot frame:

$
bold(a) = (Delta bold(p)_{t+1}, dots, Delta bold(p)_(t+H)) in RR^(H times 2)
$

Following the DDPM formulation @cite_chi2025, we learn a denoising network $epsilon_theta$ that reverses a fixed forward noising process. During training, Gaussian noise is added to expert trajectories $bold(a)_0$:

$
bold(a)_t = sqrt(overline(alpha)_t)bold(a)_0 + sqrt(1-overline(alpha)_t) epsilon
$

The model is trained to predict the noise:

$
cal(L) = EE_(t, bold(a)_0, epsilon) [ ||epsilon - epsilon_theta(bold(a)_t, t, bold(o))||^2 ]
$

=== Architecture

The denoiser $epsilon_theta$ would be implemented as a Transformer-based UNet:
- The scene latent $bold(z)_t$, robot state $bold(s)_t$, and waypoint horizon $bold(W)_t$ are projected into a shared conditioning space and injected into the UNet through cross-attention layers at multiple resolutions.
- The waypoint sequence $bold(W)_t$ is encoded temporally, enabling the policy to infer short-horizon directional intent while remaining reactive to dynamic local changes.



=== Inference

At inference time, real-time control is achieved by decoupling the high-level policy frequency from the low-level execution loop. Instead of processing every incoming frame, the Diffusion Policy operates at a lower frequency using a receding-horizon control strategy with action chunking.

At each policy step, the model observes a short temporal window of recent frames and iteratively generates a future trajectory through diffusion-based denoising:

$
bold(a)_(t-1) = 1/sqrt(alpha_t) (
bold(a)_t - (1 - alpha_t)/(sqrt(1 - overline(alpha)_t)) epsilon_theta (bold(a)_t, t, bold(o)_t)) + sigma_t bold(z)
$

Only the few first actions $Delta bold(p)_(t+1), ..., Delta bold(p)_(t+k)$ are executed, and the remainder of the horizon is discarded. The process is repeated in a receding-horizon manner, enabling continuous reactivity to dynamic changes such as pedestrian motion, group formation, or sudden occlusions.

=== Training and Social Compliance

The policy would be trained purely from demonstrations collected in populated campus environments. Each trajectory would reflect human-operated navigation in dense social settings.

No explicit social reward, handcrafted rule system, or manually designed interaction model is used. Instead, socially compliant navigation behavior emerges implicitly from the training distribution.

This data-driven formulation offers two advantages over rule-based navigation systems:
1. *Contextual generality*: social behavior in shared environments is highly contextual and continuous, making exhaustive rule specification impractical. Appropriate navigation depends jointly on geometry, crowd density, motion patterns, and approach direction.
2. *Multimodal behavior representation*: diffusion models naturally represent multiple valid future trajectories, capturing the inherent uncertainty and variability of human motion.

= Experiments

The following experiments evaluate the individual components of the proposed framework. Since the full end-to-end system integration remains future work, each experiment focuses on validating a specific subsystem independently: hierarchical scene graph construction, open-vocabulary semantic retrieval, navigation graph generation, and real-time geometry estimation.

== Experimental Setup

Although the framework ultimately targets large-scale quasi-indoor/outdoor environments, the initial scene graph construction is validated on the Habitat-Sim dataset as a controlled benchmark. Habitat-Sim provides household-scale scenes with reliable camera poses, enabling evaluation of semantic reasoning, spatial gating, and hierarchical graph construction before scaling to real-world campus environments with sensor noise and dynamic occlusions.

Experiments are conducted using the HOV-SG codebase on Habitat-Sim validation scenes, each containing roughly $2000$ RGB-D frames collected through virtual traversal. Global point clouds are reconstructed from the RGB-D streams, while semantic point clouds are generated by projecting SAM-based 2D segmentation masks into 3D space to associate semantic labels with reconstructed geometry.

Quick summary of the experiments:

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Experiments*], [*Hardware*], [*Latency*],
  ),
  [*HOV-SG* scene graph construction], [Local *NVIDIA Quadro A2000*], [> 3h],
  [*HOV-SG* natural language query], [Local *NVIDIA GTX 1650*], [7s],
  [*HOV-SG* conditioned navigation], [Local *NVIDIA GTX 1650*], [10s],
  [*VGGT* inference], [Google Colab *T4 GPU*], [1.06s]
)

== Hierarchical Scene Graph Construction

An RGB-D sequence containing approximately $2000$ frames is collected for each Habitat-Sim environment. The proposed pipeline incrementally constructs a hierarchical scene graph following the HOV-SG representation.

#columns(2, gutter: 8pt)[
  In the reproduced setup, the graph hierarchy contains three levels:
  
  $
  "Building" arrow "Room" arrow "Object"
  $

  *Results*
  
  The resulting graph successfully captures the semantic and spatial structure of household-scale environments, enabling efficient semantic retrieval and navigation reasoning.

  Object-level semantics are obtained through SAM-based segmentation combined with CLIP feature embeddings, while semantic masks are projected from image space back into the reconstructed 3D point cloud.

  #colbreak()

  #figure(
    image("images/exp_SG.png", width: 90%),
    caption: [Constructed hierarchical scene graph overlaid on the reconstructed point cloud.]
  )
]

== Open-Vocabulary Semantic Query

#columns(2, gutter: 8pt)[
  Using the previously constructed scene graph, we evaluate the system's ability to retrieve object instances from natural language descriptions. A total of $20$ queries are issued, ranging from simple category-based requests such as:
  - "find the fire extinguisher",
  - "find the monitor",
  to context-aware queries including:
  - "find the chair near the whiteboard",
  - "find the table beside the sofa".

  *Setup*

  Natural language query is also decomposed by LLM to plausible path from root to queried object. The node captions are encoded using CLIP text embeddings and matched against object node embeddings within the scene graph. The retrieved object node returns its associated semantic point cloud segment.

  In addition to semantic retrieval, random source-to-target trajectories are sampled inside the environment to validate that retrieved objects remain spatially consistent with the navigation structure.
  
  The original HOV-SG paper reports approximately $55%$ retrieval success on a comparable benchmark; we did not re-run the full quantitative evaluation. This suggests a large room for development.

  #colbreak()
  #figure(
    image("images/exp_query_resolve.png", width: 90%),
    caption: [Query Resolution]
  )
  #figure(
    image("images/exp_query.png", width: 85%),
    caption: [Retrieved point cloud for query\ *"find sink in the bathroom"*]
  )
]

#figure(
  image("images/exp_nav.png", width: 70%),
  caption: [Waypoint trajectory generated by Voronoi-based navigation graph]
)

It is important to note that while offline approaches batch-process entire video sequences—leading to high processing latencies—our proposed deployment pipeline is designed for incremental, chunk-based processing. By splitting the RGB stream into small temporal chunks and fusing features iteratively, the scene graph can be updated on the fly. This streaming approach drastically reduces effective latency and allows the system to continuously refine its semantic representation during live exploration, making the pipeline suitable for real-time robot operation.

== VGGT Geometry Estimation

#columns(2, gutter: 8pt)[
  VGGT is evaluated using casually captured RGB images taken from multiple viewpoints using an iPhone XR. Since VGGT requires multi-view inputs, inference is performed using $3$ images observing the same scene from different angles.

  Inference is executed on *Colab T4 GPU*

  *Evaluation*

  The end-to-end inference time, including the decoder stage, takes approximately 1.06 seconds.
  
  However, the proposed framework only requires the intermediate latent feature representation rather than the fully decoded geometric output. Therefore, practical deployment latency is expected to be significantly lower.

  Rather than evaluating against a LiDAR-based reconstruction pipeline, we assess the generated geometry qualitatively. 
  
  #colbreak()
  #figure(
    image("images/exp_vggt_input.png"),
    caption: [Images of my headphone, taken from multiple angle, served as input for VGGT]
  )
  #figure(
    image("images/exp_vggt_output.png"),
    caption: [Depth Map inferred by VGGT]
  )
]

Across multiple test scenes, VGGT consistently demonstrates strong geometric awareness, recovering coherent scene structure, spatial layout, and object boundaries from sparse multi-view observations, suggesting sufficiently rich geometric information for downstream local motion generation is encoded in VGGT latent representation. As a result, explicit geometric reconstruction can be delegated to the learned latent space without requiring full dense decoding during deployment.

= Limitations & Future Work

== Limitations

*System Integration*:

The framework presented in this work remains a modular proposal supported primarily through component-level experiments. Although individual subsystems — including hierarchical scene graph construction, semantic retrieval, and latent geometry estimation — demonstrate promising behavior independently, the complete pipeline has not yet been integrated into a unified embodied navigation system.

In particular, the transition from natural language query to executable local motion generation remains invalidated. As a result, the interaction between high-level semantic reasoning, global planning, and low-level motion control under real deployment conditions is still unknown.

*Diffusion policy training and adaptation.*

Rather than training from demonstrations collected from scratch, the policy can be initialized from pre-trained navigation models (NoMaD, FlowNav, Navigation World Model) and fine-tuned on a small amount of campus-specific demonstration data to close the domain gap. A systematic study of fine-tuning data efficiency — how few campus demonstrations are needed before social compliance is reliable — would be a practically valuable contribution. Flow matching @cite_lipman2022 remains a promising alternative to DDPM for reducing inference latency.

*Scene Graph Scalability*

The current experiments are conducted on household-scale environments. Although these environments provide realistic semantic indoor layouts, they remain significantly smaller and more structured than large-scale real-world campuses or urban environments, where the number of object nodes, spatial relationships, and semantic ambiguities may grow substantially, potentially affecting retrieval efficiency and graph maintenance.

*Query Decomposition Robustness*

The hierarchical query decomposition pipeline relies on LLM to generate semantically meaningful intermediate descriptions. For queries with ambiguous spatial referents or domain-specific terminology uncommon in the LLM's training data, decomposition may produce incorrect zone or area descriptions that misdirect the CLIP retrieval. No systematic evaluation of decomposition failure modes has been conducted.

*Dynamic Environments*

The spatial-temporal corridor filter for dynamic object exclusion has been adopted from CausalNav @cite_duan2026 without re-validation in our target environment. Dense pedestrian scenarios — crowded building entrances, lecture breaks — present high rates of transient detections that may stress the filter's temporal window assumptions.

*Safety Assurance and OOD Scenarios*

A known limitation of purely learning-based local motion generation is unpredictable behavior in out-of-distribution (OOD) states. Because social compliance and collision avoidance are implicitly learned rather than explicitly constrained, relying solely on a generative model for local control poses safety risks in highly dynamic campus environments. Future iterations of this framework must integrate a deterministic safety shield. A high-frequency local reactive controller (such as the Dynamic Window Approach or Control Barrier Functions) could operate continuously in the background. By utilizing low-latency geometric data, this secondary system would evaluate the diffusion-generated trajectories against immediate kinematic constraints, overriding the learned policy to execute an emergency stop or geometric evasive maneuver if an imminent collision is detected.

== Future Work

*End-to-end integration and evaluation.* The immediate next step is integrating all components into a unified system and evaluating it on a real campus navigation task. Appropriate metrics include task success rate, navigation efficiency (path length ratio), and social compliance indicators (minimum pedestrian distance, stop-and-wait frequency).

*Long-term map maintenance.* Campus environments change over time: buildings are renovated, furniture is rearranged, new structures appear. The current graph construction pipeline has no mechanism for detecting and correcting stale nodes. Incorporating a change detection module — comparing new observations against stored node embeddings and flagging divergence — would extend the framework to long-term autonomous operation.

*Guided Exploration and Onboarding Phase*. To bridge the gap between simulation and real-world campus deployment, future iterations will implement a guided exploration phase for novel environments. During this onboarding phase, the robot will traverse the target environment—such as the IP Paris campus—either via human tele-operation or by following a human guide. Because the proposed scene reconstruction pipeline relies purely on deep learning inference (SAM2, CLIP) and metric projections rather than explicit robot odometry, these continuous RGB video sequences are sufficient for initial graph construction. The transition from standard SAM to SAM2 will specifically enforce temporal consistency across these continuous video frames, stabilizing instance tracking and feature fusion during this critical first-time scene reconstruction.
  
#bibliography("works.bib", style: "ieee")

#set heading(numbering: none)

= Appendix A: Envisioned Resource Consumption Analysis

Deploying a heterogeneous foundation model stack on edge hardware—such as an NVIDIA Jetson Orin Nano Super Developer Kit—imposes strict constraints on memory bandwidth, VRAM allocation, and thermal limits. A naive deployment strategy risks severe compute under-utilization, particularly regarding the Large Language Model (LLM).

== A.1 Model Parameter Distribution

The proposed framework relies on $4$ primary DL components with distinct computational footprints:

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Model*], [*Size*], [*Role*],
  ),
  [*SAM2*], [\~1B], [dense instance segmentation and feature extraction],
  [*VGGT*], [\~1B], [local scene geometry extraction],
  [*LLM*], [4-8B], [process natural language and high-level reasoning.],
  [*CLIP*], [\~400M], [encode visual instances into a shared language-aligned embedding space for graph matching and retrieval.]
)

== A.2 The Compute Under-utilization Problem

If the system architecture restricts the LLM exclusively to the initial query decomposition task, a severe computational imbalance emerges. The LLM, consuming the single largest portion of the available VRAM budget, would execute a single forward pass to parse the user's command and then sit idle during the entire navigation run. 

Meanwhile, the smaller perception models (SAM2, VGGT), the embedding bridge (CLIP), and the local Diffusion Policy would operate continuously in a high-frequency closed loop. Allocating the heavy runtime premium of a 4B parameter model to a one-shot string parsing task represents an unacceptable bottleneck for an embodied edge agent.

== A.3 Dynamic Load Balancing and Executive Reasoning

To maximize the utility of the hardware and justify the LLM's computational footprint, its role must be expanded beyond initial query decomposition. The LLM should operate as an active, continuous global reasoning engine throughout the navigation lifecycle:

1. *Contextual Edge Weighting:* Rather than computing static shortest paths, the global planner queries the LLM to dynamically modulate the navigation graph's edge weights. The LLM applies implicit real-world knowledge to spatial routing—for example, penalizing outdoor courtyard pathways during adverse weather, or avoiding cafeteria zones during peak lunch hours.

2. *Semantic Failure Recovery:* If the local Diffusion Policy encounters an impassable, out-of-distribution state (e.g., a hallway blocked by sudden construction), it flags a local failure. The LLM is invoked to process the failure context and formulate a high-level alternative strategy (e.g., "The main corridor is blocked; reroute through the adjacent library"). 

By delegating continuous executive oversight and dynamic spatial heuristic adjustments to the LLM—while allowing CLIP, SAM2, and VGGT to handle steady-state metric perception—the framework achieves a balanced computational load, ensuring that the most resource-intensive model actively contributes to the system's robustness and adaptability.

#figure(
  image("images/appendix.png"),
)