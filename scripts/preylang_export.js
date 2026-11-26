// ============================================================
// PREY LANG FOREST CHANGE — Full Pipeline (v1.13)
// ============================================================
// Hansen v1.13: 2001–2025 loss data (lossyear values 1–25)
// Training years: 2019, 2021, 2023 (2025 excluded — preliminary layer)
// Sampling: 10,000 fixed random points, shared across all years
// Exports: 4 CSVs (training) + 16 GeoTIFF tiles (prediction/U-Net)
// ============================================================
//
// HOW TO RUN
//   1. Open https://code.earthengine.google.com
//   2. Paste this whole file into the code editor.
//   3. Click "Run", then open the "Tasks" tab and click "Run" on each export.
//   4. The files land in a Google Drive folder named "PreyLang".
//   5. Share that folder with your team (the notebook reads it from Drive).
// ============================================================

// ============================================================
// STUDY AREA (hand-drawn Prey Lang polygon)
// ============================================================
var studyArea = ee.Geometry.Polygon([[
  [105.85270402526005, 13.49624869119533],
  [105.88566300963505, 12.838357237924747],
  [105.3281068572913, 12.82496728285258],
  [105.29789445494755, 13.49624869119533],
  [105.85270402526005, 13.49624869119533]
]]);

Map.addLayer(studyArea, {color: 'red'}, 'Prey Lang (study area)');
Map.centerObject(studyArea, 9);

// ============================================================
// PREPROCESSING — cloud masking + yearly composites
// ============================================================
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3)      // cloud shadow
              .and(scl.neq(9))   // high prob cloud
              .and(scl.neq(10)); // thin cirrus
  return image.updateMask(mask).divide(10000);
}

function getYearlyComposite(year) {
  // Dry season: Dec of year-1 to Mar of year
  var startDate = ee.Date.fromYMD(year - 1, 12, 1);
  var endDate   = ee.Date.fromYMD(year, 3, 31);
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(studyArea)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
    .map(maskS2clouds)
    .median()
    .clip(studyArea);
}

// ============================================================
// LABELING — Hansen v1.13
// ============================================================
var hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13');
var treecover2000 = hansen.select('treecover2000');
var lossYear      = hansen.select('lossyear');

function makeLabelForYear(year, composite) {
  var wasForest2000 = treecover2000.gte(30);
  var yearCode = year - 2000;
  var lostByThisYear = lossYear.gt(0).and(lossYear.lte(yearCode));
  var label = wasForest2000.and(lostByThisYear.not())
              .rename('label')
              .toFloat();   // Float32 to match spectral bands
  return composite.addBands(label);
}

// ============================================================
// FIXED RANDOM POINTS — same locations across all years
// ============================================================
var N_POINTS = 10000;
var randomPoints = ee.FeatureCollection.randomPoints({
  region: studyArea,
  points: N_POINTS,
  seed: 42,
  maxError: 1
});
print('Random points created:', randomPoints.size());

// ============================================================
// EXPORT — CSVs + tiled GeoTIFFs
// ============================================================
var years = [2019, 2021, 2023, 2025];
var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'label'];

// Bounds for tiling
var bounds = studyArea.bounds().getInfo().coordinates[0];
var xmin = bounds[0][0], ymin = bounds[0][1];
var xmax = bounds[2][0], ymax = bounds[2][1];
var nTiles = 2;
var dx = (xmax - xmin) / nTiles;
var dy = (ymax - ymin) / nTiles;

years.forEach(function(y) {
  var composite = getYearlyComposite(y);
  var labeled   = makeLabelForYear(y, composite);

  // --- CSV: sample at fixed random points ---
  var samples = labeled.select(BANDS).sampleRegions({
    collection: randomPoints,
    properties: [],
    scale: 10,
    tileScale: 8,
    geometries: true
  }).map(function(f) {
    var coords = f.geometry().coordinates();
    return f.set('year', y)
            .set('lon', coords.get(0))
            .set('lat', coords.get(1));
  }).filter(ee.Filter.notNull(BANDS));

  print('Sample count for ' + y + ':', samples.size());

  Export.table.toDrive({
    collection: samples,
    description: 'PreyLang_Samples_' + y,
    folder: 'PreyLang',
    fileFormat: 'CSV',
    selectors: BANDS.concat(['year', 'lon', 'lat'])
  });

  // --- GeoTIFF: tiled composites ---
  for (var i = 0; i < nTiles; i++) {
    for (var j = 0; j < nTiles; j++) {
      var tile = ee.Geometry.Rectangle([
        xmin + i*dx, ymin + j*dy,
        xmin + (i+1)*dx, ymin + (j+1)*dy
      ]);
      Export.image.toDrive({
        image: labeled.select(BANDS).clip(tile),
        description: 'PreyLang_Composite_' + y + '_tile_' + i + '_' + j,
        folder: 'PreyLang',
        scale: 10,
        region: tile,
        maxPixels: 1e13,
        fileFormat: 'GeoTIFF'
      });
    }
  }
});
