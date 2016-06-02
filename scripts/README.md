## How to use the fetch\_sdss.py script

1. Use [CasJobs](http://skyserver.sdss.org/casjobs/) to execute the following query.
  ```sql
  SELECT spec.specObjID, phot.objID,
      spec.ra, spec.dec,
      spec.class,
      spec.z, spec.zErr,
      phot.rerun, phot.run, phot.camcol, phot.field,
      phot.dered_u, phot.dered_g, phot.dered_r, phot.dered_i, phot.dered_z,
      phot.psfMag_u, phot.psfMag_g, phot.psfMag_r, phot.psfMag_i, phot.psfMag_z,
      cModelMag_u, cModelMag_g, cModelMag_r, cModelMag_i, cModelMag_z
  INTO mydb.DR12_spec_phot_all
  FROM SpecObjAll AS spec
  JOIN PhotoObjAll AS phot
  ON spec.specObjID = phot.specObjID
  WHERE
      phot.clean = 1
      AND spec.zWarning = 0
      AND phot.dered_r > 0 and phot.dered_r < 40
      AND phot.expRad_r < 30
      AND phot.deVRad_r < 30
      AND spec.zErr < 0.1
      AND spec.z < 2
  ```
  This will create a table named `DR12_spec_phot_all` in `mydb`.

2. Select a random subset from the previous table. For example, if you want a
  random sample of 100,000 objects, execute the following.
  ```sql
  SELECT TOP 100000 *
  INTO mydb.DR12_spec_phot_sample
  FROM mydb.DR12_spec_phot_all
  ORDER BY NEWID()
  ```
  This will create a table named `DR12_spec_phot_sample` with 100,000 rows.
  Note that `ORDER BY NEWID()` randomly shuffles the rows.

3. Click `MyDB` in the menu. Select the table `DR12_spec_phot_sample`.
  Download the table as a CSV file, and save it in the same directory as
  `fetch_sdss.py`.

4. Run `fetch_sdss.py`. If you are using Docker, run e.g.
  ```shell
  $ docker run -d --name fetch -v /path/to/dl4astro/scripts:/scripts dl4astro sh -c 'cd /scripts && ./fetch_sdss.py'
  ```
  This will create a directory named `result` under `/path/to/dl4astro/scripts`
  with an npy file for each cutout image.
