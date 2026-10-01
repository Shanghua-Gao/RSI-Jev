# Image credits and licences

The pictures under `images/` are **not** covered by this repository's MIT licence. Each
one keeps its own licence, listed below. Every picture the gallery uses ships here; none
has to be downloaded.

## images/generated/ (30 files): ours, CC0

Charts, screens, receipts, tic-tac-toe boards, a food chain and shape-counting scenes,
drawn by `make_images.py` for this gallery. They are not taken from any training or
evaluation set. Released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## images/photos/ (15 files): Wikimedia Commons

These are the sample photos of [reinhard-z/vision-jev](https://github.com/reinhard-z/vision-jev)
(MIT, commit `d559c10`), which downscaled each Wikimedia Commons file to at most 512 px on
the long edge and made no other change. We ship them as that repository has them.

| file | original | author | licence |
|---|---|---|---|
| adult.jpg | [Pedestrians using crosswalk (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Pedestrians_using_crosswalk_(Unsplash).jpg) | Peter Miranda | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| amber.jpg | [Amber traffic signal, Stamford Road, Singapore - 20111210-02.jpg](https://commons.wikimedia.org/wiki/File:Amber_traffic_signal,_Stamford_Road,_Singapore_-_20111210-02.jpg) | Jacklee | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) (share-alike) |
| bag.jpg | [Discarded Meijer plastic bag.jpg](https://commons.wikimedia.org/wiki/File:Discarded_Meijer_plastic_bag.jpg) | Visviva | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| bicycle.jpg | [Mountain Bike In The Snow.jpg](https://commons.wikimedia.org/wiki/File:Mountain_Bike_In_The_Snow.jpg) | Ingolfson | Public domain |
| box.jpg | [Cardboard box.jpg](https://commons.wikimedia.org/wiki/File:Cardboard_box.jpg) | MrBeastRapper | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) (share-alike) |
| car.jpg | [Public domain image - Peugeot iOn electric car in front of wind turbines.JPG](https://commons.wikimedia.org/wiki/File:Public_domain_image_-_Peugeot_iOn_electric_car_in_front_of_wind_turbines.JPG) | Kiwiev | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| cat.jpg | [Black and white cat sitting on lawn.jpg](https://commons.wikimedia.org/wiki/File:Black_and_white_cat_sitting_on_lawn.jpg) | Grendelkhan | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) (share-alike) |
| child.jpg | [Boy and Ball (8257295231).jpg](https://commons.wikimedia.org/wiki/File:Boy_and_Ball_(8257295231).jpg) | Evgeniy Isaev | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| dog.jpg | [Boxer dog posing on the grass.jpg](https://commons.wikimedia.org/wiki/File:Boxer_dog_posing_on_the_grass.jpg) | Joselodos | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |
| green.jpg | [Green traffic signal, Stamford Road, Singapore - 20111210-03.jpg](https://commons.wikimedia.org/wiki/File:Green_traffic_signal,_Stamford_Road,_Singapore_-_20111210-03.jpg) | Jacklee | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) (share-alike) |
| leaves.jpg | [Laubdeponie, Gartenanlage Steinhof, Wien.JPG](https://commons.wikimedia.org/wiki/File:Laubdeponie,_Gartenanlage_Steinhof,_Wien.JPG) | Wald1siedel | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) (share-alike) |
| limit80.jpg | [Korean Sign - Maximum Speed Limit 80kph 1.jpg](https://commons.wikimedia.org/wiki/File:Korean_Sign_-_Maximum_Speed_Limit_80kph_1.jpg) | P.Ctnt | Public domain |
| red.jpg | [Red traffic signal, Stamford Road, Singapore - 20111210-01.jpg](https://commons.wikimedia.org/wiki/File:Red_traffic_signal,_Stamford_Road,_Singapore_-_20111210-01.jpg) | Jacklee | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) (share-alike) |
| stop.jpg | [Stop sign us.jpg](https://commons.wikimedia.org/wiki/File:Stop_sign_us.jpg) | Dori | Public domain |
| teddy.jpg | [Heliograph of Bear Bear Bear Bear Bear Bear Bear Bear Bear Bear Bear Bear Bear .png](https://commons.wikimedia.org/wiki/File:Heliograph_of_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_Bear_.png) | Pixabay | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |

**Share-alike.** amber.jpg, green.jpg, red.jpg, leaves.jpg (CC BY-SA 3.0), cat.jpg and box.jpg
(CC BY-SA 4.0) are downscaled adaptations of the originals. If you redistribute them, or
anything you make from them (a crop, a card, a frame of a video), credit the author as
above and release that work under the same licence. The other photos have no such condition.

vision-jev's samples also include `limit30.jpg`. Its source is not recorded, so it is not
shipped here and no question uses it.

## images/clevr/ (38 files): CLEVR, CC BY 4.0

From CLEVR v1.0 (Johnson et al., "CLEVR: A Diagnostic Dataset for
Compositional Language and Elementary Visual Reasoning", CVPR 2017), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Taken from the `clevr` subset of
[HuggingFaceM4/the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
(revision `847a98a7`) and re-saved as JPEG (quality 90); the file name is the row number in
that subset. Credit: "CLEVR, Johnson et al. 2017, CC BY 4.0".

## images/visa/ (6 files): VisA, CC BY 4.0

Unmodified photos from the VisA dataset (Zou et al., "SPot-the-Difference Self-Supervised
Pre-training for Anomaly Detection and Segmentation", ECCV 2022; Amazon Science), archive
`VisA_20220922.tar`, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Renamed from `<category>/Data/Images/<Normal|Anomaly>/<n>.JPG` to `<category>_<Normal|Anomaly>_<n>.JPG`.
Credit: "VisA dataset, Zou et al. 2022, CC BY 4.0".
