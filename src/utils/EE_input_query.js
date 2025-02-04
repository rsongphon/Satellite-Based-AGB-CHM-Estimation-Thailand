/////////////////// User defined variables ////////////////////////////

// Country boundary polygon
var target = target_shapefile



// Start and end date to filter collections

var startDate = '2018-11-01'; //YYY-MM-DD 
var endDate = '2019-03-30';   //YYY-MM-DD 


// Maximum cloud cover percentage
var cloudCoverPerc = 50;


///////////////////////////////////////////////////////////////////////


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
var s2Filt = s2.filterBounds(target)
                .filterDate(startDate,endDate)
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',cloudCoverPerc)
                .map(maskS2srClouds);



// Composite images
var s2composite = s2Filt.median().clip(target); // can be changed to mean, min, etc


var exportImage = s2composite.select('B.*').toFloat();


///////////////////////////////////////////////////////////////////////

// ----------------- SENTINEL-1 COLLECTION ------------------------
// Sentinel 1 Descending Image Is Out of Operation since December 2021!!

/// VV
var imgVV = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(target)
        .filterDate(startDate,endDate)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
        .map(function(image) {
          var edgeleft = image.lt(-30.0);
          var edgeright = image.gt(0.0);
          var maskedImage = image.mask().and(edgeleft.not()).and(edgeright.not());
          return image.updateMask(maskedImage);
        });
        

var desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var asc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var descVVimg = desc.median().clip(target).add(30).divide(30).toFloat();
var ascVVimg = asc.median().clip(target).add(30).divide(30).toFloat();
var neighborhoodSize = 1; // Adjust this value according to your needs

// Select Ascending Image
var ascVVFillingimg = ascVVimg.focal_mean({radius: neighborhoodSize, units: 'pixels', kernelType: 'circle'});




/// VH
var imgVH = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(target)
        .filterDate(startDate,endDate)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VH')
        .map(function(image) {
          var edgeleft = image.lt(-30.0);
          var edgeright = image.gt(0.0);
          var maskedImage = image.mask().and(edgeleft.not()).and(edgeright.not());
          return image.updateMask(maskedImage);
        });


var desc = imgVH.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var asc = imgVH.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var descVHimg = desc.median().clip(target).add(30).divide(30).toFloat();
var ascVHimg = asc.median().clip(target).add(30).divide(30).toFloat();
var neighborhoodSize = 1; // Adjust this value according to your needs

// Select Ascending Image
var ascVHFillingimg = ascVHimg.focal_mean({radius: neighborhoodSize, units: 'pixels', kernelType: 'circle'});



///////////////////////////////////////////////////////////////////////

// ----------------- Export Product ------------------------

// Create a multiband composite image
var composite = ee.Image([exportImage, ascVVimg, ascVHimg])


    Export.image.toCloudStorage({
    image: composite,
    description: 'Sen1_Sen2_Thailand_Median_Composite_Raw_10m_32647_14layer_predict_2019',
    //description: 'Sen1_Sen2_Sak_Median_Composite_Raw_10m_32647_14layer_predict',
    bucket: 'varuna-data-nonprod-analytic',
    fileNamePrefix: 'biomass-estimation-project/vm-backup/AGB_model_data/raw_2019/Sen1_Sen2_Median_Composite_Raw_10m_32647_14layer_predcit_large_2019',
    //fileNamePrefix: 'biomass-estimation-project/vm-backup/Canopy_model_data/Compostie_image_raw_predict/Sen1_Sen2_Sak_Median_Composite_Raw_10m_32647_14layer_predcit',
    scale: 50,
    region: target,
    maxPixels: 1e13
  });