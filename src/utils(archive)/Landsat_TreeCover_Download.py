
import requests
import bs4
userid = 'songphon'
password = '?,?xU97y8##VUSu'

# Generate rath and row number for Thailand Landsat data

p = [str(i) for i in range(126,133)]
r = ['0'+str(i) for i in range(46,57)]
'''
# 2000 Data

site_url = 'https://e4ftl01.cr.usgs.gov/MEASURES/GFCC30TC.003/2000.01.01'
# create session
s = requests.Session()
# GET request. 
s.get(site_url)
s.post(site_url, data={'_username': userid, '_password': password})


for row in r:

    file_url = f'https://e4ftl01.cr.usgs.gov/MEASURES/GFCC30TC.003/2000.01.01/GFCC30TC_p'+'132'+'r'+row+'_TC_2000.zip'
    o_file = f'/home/jupyter/Sentinel2_Data/Landsat_CanopyCover/2000/GFCC30TC_p'+'132'+'r'+row+'_TC_2000.zip'  

    # Visit url
    r = s.get(file_url)
    # Download file
    with open(o_file, 'wb') as output:
        output.write(r.content)
    print(f"requests:: File {o_file} downloaded successfully!")

# Close session once all work done
s.close()
print('All 2000 Data Download Done')
'''
site_url = 'https://e4ftl01.cr.usgs.gov/MEASURES/GFCC30TC.003/2010.01.01'
# create session
s = requests.Session()
# GET request. 
s.get(site_url)
s.post(site_url, data={'_username': userid, '_password': password})


for row in r:
    file_url = f'https://e4ftl01.cr.usgs.gov/MEASURES/GFCC30TC.003/2010.01.01/GFCC30TC_p'+'132'+'r'+row+'_TC_2010.zip'
    o_file = f'/home/jupyter/Sentinel2_Data/Landsat_CanopyCover/2010/GFCC30TC_p'+'132'+'r'+row+'_TC_2010.zip'  

    # Visit url
    r = s.get(file_url)

    # Download file
    with open(o_file, 'wb') as output:
        output.write(r.content)
    print(f"requests:: File {o_file} downloaded successfully!")

# Close session once all work done
s.close()

print('All 2010 Data Download Done')
