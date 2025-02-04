// import target shapefile in earth engine environment

var country = target_shp.geometry();

// define temporal range
var startDate = '2020-11-01'; //YYY-MM-DD
var endDate = '2021-04-30'; // YYY-MM-DD

////// Query Sentinel 2 to create NDVI image ////////////


// Country  Name
var countryName = 'Thailand';

// Maximum cloud cover percentage
var cloudCoverPerc = 50;
// ----------------- SENTINEL-2 COLLECTION ------------------------

// We will use the Sentinel-2 Surface Reflection product.
// This dataset has already been atmospherically corrected
var s2 = ee.ImageCollection("COPERNICUS/S2_SR");

// Function to mask clouds S2
function maskS2srClouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Filter Sentinel-2 collection
var s2Filt = s2.filterBounds(country)
                .filterDate(startDate,endDate)
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',cloudCoverPerc)
                .map(maskS2srClouds);

///// NDVI /////
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
};

var s2withNDVI = s2Filt.map(addNDVI);

// Max Annual NDVI 

var s2MaxNDVIComposite = s2withNDVI.select('NDVI').max();


// NDVI mask

var NDVIMask03 = s2MaxNDVIComposite.lt(0.3);
var NDVIMask05 = s2MaxNDVIComposite.lt(0.5);



// ESA World Cover Filter 
var ESA = ee.ImageCollection('ESA/WorldCover/v200').first();
var ESA_thai = ESA.clip(country);
// Create mask to filter GEDI
// Not equal 50 60 70 90 100
var shrub_land_mask = ESA_thai.eq(20);
var grass_land_mask = ESA_thai.eq(30);
var crop_land_mask = ESA_thai.eq(40);
var build_up_mask = ESA_thai.eq(50);
var sparse_veg_mask = ESA_thai.eq(60);
var snow_mask = ESA_thai.eq(70);
var water_mask = ESA_thai.eq(80);
var wetland_mask = ESA_thai.eq(90);
var moss_mask = ESA_thai.eq(100);

// Tree cover
var treecovdataset2000 = ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3')
                  .filter(ee.Filter.date('2000-01-01', '2000-12-31'));
var treeCanopyCover2000 = treecovdataset2000.select('tree_canopy_cover');
var treeCanopyCover2000Mean = treeCanopyCover2000.mean();


var treecovdataset2015 = ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3')
                  .filter(ee.Filter.date('2015-01-01', '2015-12-31'));
var treeCanopyCover2015 = treecovdataset2015.select('tree_canopy_cover');
var treeCanopyCover2015Mean = treeCanopyCover2015.mean();

// Tree Cover mask
var treeCanopyCover2000mask = treeCanopyCover2000Mean.lt(20);
var treeCanopyCover2015mask = treeCanopyCover2015Mean.lt(20);
var treeCanopyCover2000mask2 = treeCanopyCover2000Mean.gte(200);
var treeCanopyCover2015mask2 = treeCanopyCover2015Mean.gte(200);

// DEM 
var SRTM = ee.Image('CGIAR/SRTM90_V4');
var elevation = SRTM.select('elevation').clip(country);
var slope = ee.Terrain.slope(elevation);

var slope_mask = slope.lt(30);

var qualityMask = function(im) {
  return im.updateMask(im.select('quality_flag').eq(1))
      .updateMask(im.select('degrade_flag').eq(0));
};

var slopeMask = function(im) {
  return im.updateMask(slope_mask);
};

///// mask
/// Tree cover < 20
//1.Pixel is classify as water (ESA == 80)
var water_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(water_mask);
//2. Pixel locate in urban area and Maximum annual NDVI below 0.5
// 50 Built up
var buildup_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(build_up_mask).and(NDVIMask05);
// 60 Sparse Vegetation
var sparse_veg_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(sparse_veg_mask).and(NDVIMask05);
// 40 Crop land
var crop_land_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(crop_land_mask).and(NDVIMask05);
// 30 Grass land
var grass_land_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(grass_land_mask).and(NDVIMask05);
// 20 Shrub land
var shrub_land_pixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(shrub_land_mask).and(NDVIMask05);
// below 0.3 NDVI
var ndvipixel = treeCanopyCover2000mask.and(treeCanopyCover2015mask).and(NDVIMask03);

//Tree cover >= 200
//1.Pixel is classify as water (ESA == 80)
var water_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(water_mask);
//2. Pixel locate in urban area and Maximum annual NDVI below 0.5
// 50 Built up
var buildup_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(build_up_mask).and(NDVIMask05);
// 60 Sparse Vegetation
var sparse_veg_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(sparse_veg_mask).and(NDVIMask05);
// 40 Crop land
var crop_land_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(crop_land_mask).and(NDVIMask05);
// 30 Grass land
var grass_land_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(grass_land_mask).and(NDVIMask05);
// 20 Shrub land
var shrub_land_pixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(shrub_land_mask).and(NDVIMask05);
// below 0.3 NDVI
var ndvipixel2 = treeCanopyCover2000mask2.and(treeCanopyCover2015mask2).and(NDVIMask03);

var mapNonCanopyValue = function(image) {
  var remapimage = image.where(image.select('rh95').lt(3) , 0.0)
                        .where(water_pixel , 0.0)
                        .where(buildup_pixel , 0.0)
                        .where(sparse_veg_pixel , 0.0)
                        .where(crop_land_pixel , 0.0)
                        .where(grass_land_pixel , 0.0)
                        .where(shrub_land_pixel , 0.0)
                        .where(ndvipixel , 0.0)
                        .where(water_pixel2 , 0.0)
                        .where(buildup_pixel2 , 0.0)
                        .where(sparse_veg_pixel2 , 0.0)
                        .where(crop_land_pixel2 , 0.0)
                        .where(grass_land_pixel2 , 0.0)
                        .where(shrub_land_pixel2 , 0.0)
                        .where(ndvipixel2 , 0.0);
  return remapimage;
};


var dataset = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
                  .filterBounds(country)
                  .filterDate(startDate,endDate)
                  .map(qualityMask)
                  .select('rh95')
                  .map(slopeMask)
                  .map(mapNonCanopyValue);


var gedi_all = dataset.mean();


Map.addLayer(gedi_all, {min: 0, max: 100}, 'Gedi L2A', true);


// Export to cloud storage
Export.image.toCloudStorage({
    image: gedi_all.toDouble(),
    description: 'GEDI_L2A_32647_10m',
    bucket: 'varuna-data-nonprod-analytic',
    fileNamePrefix: 'biomass-estimation-project/vm-backup/Canopy_model_data/GEDI_L2A/GEDI_L2A_32647_10m',
    scale: 10,
    region: country,
    maxPixels: 1e13
      });
      
    Export.image.toCloudStorage({
    image: ESA_thai,
    description: 'ESA_thai_10m',
    bucket: 'varuna-data-nonprod-analytic',
    fileNamePrefix: 'biomass-estimation-project/vm-backup/Canopy_model_data/ESA/ESA_thai_10m',
    scale: 10,
    region: country,
    maxPixels: 1e13
      });
    