import os
import datetime
import gc
import glob
import snappy
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from snappy import ProductIO

class sentinel1_download_preprocess():
    def __init__(self, input_dir, date_1, date_2, query_style, footprint, lat=24.84, lon=90.43, download=False):
        self.input_dir = input_dir
        self.date_start = datetime.datetime.strptime(date_1, "%d%b%Y")
        self.date_end = datetime.datetime.strptime(date_2, "%d%b%Y")
        self.query_style = query_style
        self.footprint = geojson_to_wkt(read_geojson(footprint))
        self.lat = lat
        self.lon = lon
        self.download = download

        # configurations
        self.api = SentinelAPI('bioman360', 'ANVjY4ae@4Gf9kY', 'https://scihub.copernicus.eu/dhus')
        self.producttype = 'GRD'  # SLC, GRD, OCN
        self.orbitdirection = 'ASCENDING'  # ASCENDING, DESCENDING
        self.sensoroperationalmode = 'IW'  # SM, IW, EW, WV

    def sentinel1_download(self):
        global download_candidate
        if self.query_style == 'coordinate':
            download_candidate = self.api.query('POINT({0} {1})'.format(self.lon, self.lat),
                                                date=(self.date_start, self.date_end),
                                                producttype=self.producttype,
                                                orbitdirection=self.orbitdirection,
                                                sensoroperationalmode=self.sensoroperationalmode)
        elif self.query_style == 'footprint':
            download_candidate = self.api.query(self.footprint,
                                                date=(self.date_start, self.date_end),
                                                producttype=self.producttype,
                                                orbitdirection=self.orbitdirection,
                                                sensoroperationalmode=self.sensoroperationalmode)
        else:
            print("Define query attribute")

        title_found_sum = 0
        for key, value in download_candidate.items():
            for k, v in value.items():
                if k == 'title':
                    title_info = v
                    title_found_sum += 1
                elif k == 'size':
                    print("title: " + title_info + " | " + v)
        print("Total found " + str(title_found_sum) +
              " title of " + str(self.api.get_products_size(download_candidate)) + " GB")

        os.chdir(self.input_dir)
        if self.download:
            if glob.glob(input_dir + "*.zip") not in [value for value in download_candidate.items()]:
                self.api.download_all(download_candidate)
                print("Nothing to download")
        else:
            print("Escaping download")
        # proceed processing after download is complete
        self.sentinel1_preprocess()

    def sentinel1_preprocess(self):
        # Get snappy Operators
        snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
        # HashMap Key-Value pairs
        HashMap = snappy.jpy.get_type('java.util.HashMap')

        for folder in glob.glob(self.input_dir + "\*"):
            gc.enable()
            if folder.endswith(".zip"):
                timestamp = folder.split("_")[5]
                sentinel_image = ProductIO.readProduct(folder)
                if self.date_start <= datetime.datetime.strptime(timestamp[:8], "%Y%m%d") <= self.date_end:
                    # add orbit file
                    self.sentinel1_preprocess_orbit_file(timestamp, sentinel_image, HashMap)
                    # remove border noise
                    self.sentinel1_preprocess_border_noise(timestamp, HashMap)
                    # remove thermal noise
                    self.sentinel1_preprocess_thermal_noise_removal(timestamp, HashMap)
                    # calibrate image to output to Sigma and dB
                    self.sentinel1_preprocess_calibration(timestamp, HashMap)
                    # TOPSAR Deburst for SLC images
                    if self.producttype == 'SLC':
                        self.sentinel1_preprocess_topsar_deburst_SLC(timestamp, HashMap)
                    # multilook
                    self.sentinel1_preprocess_multilook(timestamp, HashMap)
                    # subset using a WKT of the study area
                    self.sentinel1_preprocess_subset(timestamp, HashMap)
                    # finally terrain correction, can use local data but went for the default 
                    self.sentinel1_preprocess_terrain_correction(timestamp, HashMap)
                    # break # try this if you want to check the result one by one
            
    def sentinel1_preprocess_orbit_file(self, timestamp, sentinel_image, HashMap):
        start_time_processing = datetime.datetime.now()
        orb = self.input_dir + "\\orb_" + timestamp

        if not os.path.isfile(orb + ".dim"):
            parameters = HashMap()
            orbit_param = snappy.GPF.createProduct("Apply-Orbit-File", parameters, sentinel_image)
            ProductIO.writeProduct(orbit_param, orb, 'BEAM-DIMAP')  # BEAM-DIMAP, GeoTIFF-BigTiff
            print("orbit file added: " + orb +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + orb)

    def sentinel1_preprocess_border_noise(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        border = self.input_dir + "\\bordr_" + timestamp

        if not os.path.isfile(border + ".dim"):
            parameters = HashMap()
            border_param = snappy.GPF.createProduct("Remove-GRD-Border-Noise", parameters,
                                                    ProductIO.readProduct(self.input_dir +
                                                                          "\\orb_" + timestamp + ".dim"))
            ProductIO.writeProduct(border_param, border, 'BEAM-DIMAP')
            print("border noise removed: " + border +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + border)

    def sentinel1_preprocess_thermal_noise_removal(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        thrm = self.input_dir + "\\thrm_" + timestamp

        if not os.path.isfile(thrm + ".dim"):
            parameters = HashMap()
            thrm_param = snappy.GPF.createProduct("ThermalNoiseRemoval", parameters,
                                                  ProductIO.readProduct(self.input_dir + "\\bordr_" +
                                                                        timestamp + ".dim"))
            ProductIO.writeProduct(thrm_param, thrm, 'BEAM-DIMAP')
            print("thermal noise removed: " + thrm +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + thrm)

    def sentinel1_preprocess_calibration(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        calib = self.input_dir + "\\calib_" + timestamp

        if not os.path.isfile(calib + ".dim"):
            parameters = HashMap()
            parameters.put('outputSigmaBand', True)
            parameters.put('outputImageScaleInDb', False)
            calib_param = snappy.GPF.createProduct("Calibration", parameters,
                                                   ProductIO.readProduct(self.input_dir + "\\thrm_" +
                                                                         timestamp + ".dim"))
            ProductIO.writeProduct(calib_param, calib, 'BEAM-DIMAP')
            print("calibration complete: " + calib +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + calib)

    def sentinel1_preprocess_topsar_deburst_SLC(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        deburst = self.input_dir + "\\dburs_" + timestamp

        if not os.path.isfile(deburst):
            parameters = HashMap()
            parameters.put('outputSigmaBand', True)
            parameters.put('outputImageScaleInDb', False)
            deburst_param = snappy.GPF.createProduct("TOPSAR-Deburst", parameters,
                                                     ProductIO.readProduct(self.input_dir + "\\calib_" +
                                                                           timestamp + ".dim"))
            ProductIO.writeProduct(deburst_param, deburst, 'BEAM-DIMAP')
            print("deburst complete: " + deburst +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + deburst)

    def sentinel1_preprocess_multilook(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        multi = self.input_dir + "\\multi_" + timestamp

        if not os.path.isfile(multi + ".dim"):
            parameters = HashMap()
            parameters.put('outputSigmaBand', True)
            parameters.put('outputImageScaleInDb', False)
            multi_param = snappy.GPF.createProduct("Multilook", parameters,
                                                   ProductIO.readProduct(self.input_dir + "\\calib_" +
                                                                         timestamp + ".dim"))
            ProductIO.writeProduct(multi_param, multi, 'BEAM-DIMAP')
            print("multilook complete: " + multi +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + multi)

    def sentinel1_preprocess_subset(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        subset = self.input_dir + "\\subset_" + timestamp

        if not os.path.isfile(subset + ".dim"):
            WKTReader = snappy.jpy.get_type('com.vividsolutions.jts.io.WKTReader')
            
            # converting shapefile to GEOJSON and WKT is easy with any free online tool
            wkt = "POLYGON((92.330290184197 20.5906091141114,89.1246637610338 21.6316051481971," \
                  "89.0330319081811 21.7802436586492,88.0086282580443 24.6678836192818,88.0857830091018 " \
                  "25.9156771178278,88.1771488779853 26.1480664053835,88.3759125970998 26.5942658997298," \
                  "88.3876586919721 26.6120432770312,88.4105534167129 26.6345128356038,89.6787084683935 " \
                  "26.2383305017275,92.348481691233 25.073636976939,92.4252199249342 25.0296592837972," \
                  "92.487261172615 24.9472465376954,92.4967290851295 24.902213855393,92.6799861774377 " \
                  "21.2972058618174,92.6799346581579 21.2853347419811,92.330290184197 20.5906091141114))"

            geom = WKTReader().read(wkt)
            parameters = HashMap()
            parameters.put('geoRegion', geom)
            subset_param = snappy.GPF.createProduct("Subset", parameters,
                                                    ProductIO.readProduct(self.input_dir + "\\multi_" +
                                                                          timestamp + ".dim"))
            ProductIO.writeProduct(subset_param, subset, 'BEAM-DIMAP')
            print("subset complete: " + subset +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + subset)

    def sentinel1_preprocess_terrain_correction(self, timestamp, HashMap):
        start_time_processing = datetime.datetime.now()
        terr = self.input_dir + "\\terr_" + timestamp

        if not os.path.isfile(terr + ".dim"):
            parameters = HashMap()
            # parameters.put('demResamplingMethod', 'NEAREST_NEIGHBOUR')
            # parameters.put('imgResamplingMethod', 'NEAREST_NEIGHBOUR')
            # parameters.put('pixelSpacingInMeter', 10.0)
            terr_param = snappy.GPF.createProduct("Terrain-Correction", parameters,
                                                  ProductIO.readProduct(self.input_dir + "\\subset_" +
                                                                        timestamp + ".dim"))
            ProductIO.writeProduct(terr_param, terr, 'BEAM-DIMAP')
            print("terrain corrected: " + terr +
                  " | took: " + str(datetime.datetime.now() - start_time_processing).split('.', 2)[0])
        else:
            print("file exists - " + terr)

input_dir = "/home/jupyter/gcs/biomass-estimation-project/vm-backup/Sentinel1_Data/47PQQ"
start_date = '01Nov2020'
end_date = '30Apr2021'
query_style = 'footprint' # 'footprint' to use a GEOJSON, 'coordinate' to use a lat-lon 
footprint = '/home/jupyter/boundary_47PQQ_latlon.geojson'
lat = 26.23
lon = 88.56

sar = sentinel1_download_preprocess(input_dir, start_date, end_date, query_style, footprint, lat, lon, True) 
# proceed to download by setting 'True', default is 'False'
sar.sentinel1_download()
