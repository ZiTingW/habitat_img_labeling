# Habitat img labeling via CLIP w/o training

## Providing Example Images for Habitat Labeling

To enable quick zero-shot habitat labeling using CLIP with prompt and visual guidance, you can manually provide a few example images for each class. These should be organized under the `substrate_protos` folder, with each subdirectory named by the class index.

![](https://github.com/ZiTingW/habitat_img_labeling/blob/main/habitat_labeling.png)

### Directory structure example:
        
Manually provide some img examples to the path (substrate_protos), such as 
```
substrate_protos/
├── 0/
│   └── img_cls0_1.jpg
├── 1/
│   ├── img_cls1_1.jpg
│   └── img_cls1_2.jpg
└── ...
```

Each folder (e.g., `0/`, `1/`, etc.) corresponds to a class label. The images inside are used as visual prototypes alongside textual prompts to assign labels to new samples.

## Providing Textual Prompts

Define a dictionary substrate_prompts where each class index maps to a list of descriptive phrases:
substrate_prompts = 

{

    0: [
        "bare sandy seabed with absolutely no visible biota",
        "clean sand bottom with no visible benthic organisms",
        "featureless sandy seafloor",
        "desert-like underwater sand with no biota"
    ],

    1: [
        "evenly scattered sessile invertebrates on the seafloor",
        "sparse but consistent distribution of benthic fauna",
        "light coverage of sponges or invertebrates",
        "uniform low-density benthic community"
    ],

    2: [
        "seafloor with patchy distribution of benthic organisms",
        "mixed coverage with dense areas and bare patches",
        "transition zone from dense to sparse benthic coverage",
        "heterogeneous distribution of sessile invertebrates"
    ],

    3: [
        "dense aggregation of sessile benthic organisms",
        "highly covered seafloor with sponges and corals",
        "rich and continuous benthic invertebrate community",
        "extensive bottom coverage by filter feeders or reef fauna"
    ]
}

These prompts will be combined with visual prototypes and passed to the CLIP model for similarity-based label assignment.
