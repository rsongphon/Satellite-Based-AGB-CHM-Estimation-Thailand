// import target shapefile in earth engine environment

var country = target_shp.geometry();

// define temporal range
var startDate = '2021-11-01'; //YYY-MM-DD
var endDate = '2022-04-30'; // YYY-MM-DD

// ESA World Cover Filter 
var ESA = ee.ImageCollection('ESA/WorldCover/v200').first();
var ESA_thai = ESA.clip(country)

// Classification Mask

var build_up_mask = ESA_thai.neq(50);
var sparse_veg_mask = ESA_thai.neq(60);
var snow_mask = ESA_thai.neq(70);
var water_mask = ESA_thai.neq(80);
var wetland_mask = ESA_thai.neq(90);
var moss_mask = ESA_thai.neq(100);
var all_mask = build_up_mask.and(sparse_veg_mask).and(snow_mask).and(water_mask).and(wetland_mask).and(moss_mask)
var esaMask = function(im) {
  return im.updateMask(all_mask);
};

// DEM -> Slope Mask
var SRTM = ee.Image('CGIAR/SRTM90_V4');
var elevation = SRTM.select('elevation').clip(country);
var slope = ee.Terrain.slope(elevation);

var slope_mask = slope.lt(30);

var slopeMask = function(im) {
  return im.updateMask(slope_mask);
};

// GEDI Qualityi filter Mask
var qualityMask = function(im) {
  return im.updateMask(im.select('l4_quality_flag').eq(1))
      .updateMask(im.select('degrade_flag').eq(0))
      .updateMask(im.select('agbd_se').divide(im.select('agbd')).multiply(100).lt(50));
};


// Dataset query

var dataset = ee.ImageCollection('LARSE/GEDI/GEDI04_A_002_MONTHLY')
                  .filterBounds(country)
                  .filterDate(startDate,endDate)
                  .map(qualityMask)
                  .map(esaMask)
                  .map(slopeMask)
                  .select('agbd');


// mean composite 
var gedi_all = dataset.mean();

// Export data to gcs
Export.image.toCloudStorage({
    image: gedi_all,
    description: 'GEDI_L4A_25m_2021_2022',
    bucket: 'varuna-data-nonprod-analytic',
    fileNamePrefix: 'biomass-estimation-project/vm-backup/AGB_model_data/GEDI_L4A_25m_2021_2022',
    scale: 25,
    region: country,
    maxPixels: 1e13
  });

