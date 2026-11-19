# Download the pinned Hugging Face data (byte-identical for every teammate).
#   composites -> data/hf/composites/   (the 4 yearly images)
#   tiles      -> data/tiles/           (the 484 training chips)
$rev = "44ae7414948889ba3700f2c3b72a9fc69cffe270"

hf download umeshuu/cambodia-deforestation --repo-type dataset `
  --revision $rev --include "composites/*" --local-dir data/hf

hf download umeshuu/cambodia-deforestation --repo-type dataset `
  --revision $rev --include "tiles/*" --local-dir data
