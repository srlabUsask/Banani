
#@author: Amit Kumar Mondal
#@address: SR LAB, Computer Science Department, USASK
#Email: amit.mondal@usask.ca
import paramiko
from itertools import islice
import re
import sys
import os
import io
#import errno
import cv2
import math
from numpy import linalg
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext, Row
from pyspark.mllib.clustering import KMeans
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel
#import functools
from skimage import *
from skimage import color
from skimage.feature import blob_doh
from io import BytesIO
import csv
from time import time

pipeline_obj = object()
sc= object()
spark =object()
npartitions = 8
logger = ''

from scipy.misc import imread, imsave

class ImgPipeline:
    IMG_SERVER = ''
    U_NAME = ''
    PASSWORD = ''
    LOADING_PATH = ''
    SAVING_PATH = ''
    CSV_FILE_PATH = '/home/amit/segment_data/imglist.csv'
    def __init__(self, server, uname, password):
        self.IMG_SERVER = server
        self.U_NAME = uname
        self.PASSWORD = password

    def setLoadAndSavePath(self,loadpath, savepath):
        self.LOADING_PATH = loadpath
        self.SAVING_PATH = savepath

    def setCSVAndSavePath(self,csvpath, savepath):
        self.CSV_FILE_PATH = csvpath
        self.SAVING_PATH = savepath

    def matchpart(self, key):
        #print("passedkey"+ key)
        rededge_channel_pattern = re.compile('(.+)_[0-9]+')
        match = rededge_channel_pattern.search(key)
        if(match == None):
            return key
        return match.group(1)

    def collectDirs(self,apattern = '"*.jpg"'):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        ftp = ssh.open_sftp()
        apath = self.LOADING_PATH
        if(apattern=='*' ):
            apattern = '"*"'
        rawcommand = 'find {path} -name {pattern}'
        command = rawcommand.format(path=apath, pattern=apattern)
        stdin, stdout, stderr = ssh.exec_command(command)
        filelist = stdout.read().splitlines()
        print(len(filelist))
        return filelist

    def collectFiles(self, ext):
        files = self.collectDirs(ext)
        filenames = set()
        for file in files:
            if (len(file.split('.')) > 1):
                filenames.add(file)

        filenames = list(filenames)
        return filenames
    def collectImgFromCSV(self, column):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        csvfile = None
        try:
            ftp = ssh.open_sftp()

            file = ftp.file(self.CSV_FILE_PATH, "r", -1)
            buf = file.read()
            csvfile = BytesIO(buf)

            ftp.close()
        except IOError as e:
            print(e)
        # ftp.close()
        ssh.close()
        contnts = csv.DictReader(csvfile)
        filenames = set()
        for row in contnts:
            filenames.add(row[column])
        return list(filenames)

    def ImgandParamFromCSV(self, column1, column2):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        csvfile = None
        try:
            ftp = ssh.open_sftp()

            file = ftp.file(self.CSV_FILE_PATH, "r", -1)
            buf = file.read()
            csvfile = BytesIO(buf)

            ftp.close()
        except IOError as e:
            print(e)
        # ftp.close()
        ssh.close()
        contnts = csv.DictReader(csvfile)
        filenames = {}
        for row in contnts:
            filenames[row[column1]] = row[column2]
            # filenames.add(row[column])
        return filenames

    def collectImagesSet(self,ext):

        #Collect sub-directories and files of a given directory
        filelist = self.collectDirs(ext)
        print(len(filelist))
        dirs = set()
        dirs_list = []
        #Create a dictionary that contains: sub-directory --> [list of images of that directory]
        dirs_dict = dict()
        for afile in filelist:
            (head, filename) = os.path.split(afile)
            if (head in dirs):
                if (head != afile):
                    dirs_list = dirs_dict[head]
                    dirs_list.append(afile)
                    dirs_dict.update({head: dirs_list})
            else:
                dirs_list = []
                if (len(filename.split('.')) > 1 and head != afile):
                    dirs_list.append(afile)
                    dirs.add(head)
                    dirs_dict.update({head: dirs_list})

        return dirs_dict.items()

    def collectImgsAsGroup(self, file_abspaths):
        rededge_channel_pattern = re.compile('(.+)_[0-9]+\.tif$')
        # TODO merge this with the RededgeImage object created by Javier.
        image_sets = {}

        for path in file_abspaths:
            match = rededge_channel_pattern.search(path)
            if match:
                common_path = match.group(1)
                if common_path not in image_sets:
                    image_sets[common_path] = []
                image_sets[common_path].append(path)
        grouping_as_dic = dict()
        for grp in image_sets:

            grouping_as_dic.update({grp: image_sets[grp]})
        return grouping_as_dic.items()
    def splitHdfs(self, dpack):
        #print(len(dpack))
        fname = dpack[0]
        #print(fname)
        imgbuf = imread(BytesIO(dpack[1]))
        return [(fname, imgbuf)]
    def loadIntoCluster(self, path, offset=None, size=-1):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD, timeout=60)
        imagbuf = ''
        try:
            ftp = ssh.open_sftp()

            file = ftp.file(path, 'r', (-1))
            buf = file.read()
            imagbuf = imread(BytesIO(buf))

            ftp.close()
        except IOError as e:
            print(e)
        # ftp.close()
        #time.sleep()
        ssh.close()
        return (path, imagbuf)

    def loadBundleIntoCluster_Skip_conversion(self, path, offset=None, size=(-1)):

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        images = []
        sortedpath= path[1]
        sortedpath.sort()
        print(sortedpath)
        try:
            for img_name in sortedpath:
                ftp = ssh.open_sftp()
                file = ftp.file(img_name, 'r', (-1))
                buf = file.read()
                imagbuf = imread(BytesIO(buf))
                if(imagbuf != None) and (hasattr(imagbuf, 'shape')):
                    images.append(imagbuf)
                else:
                    print("Warning: corrupted image skipped -->" + img_name )
                ftp.close()
        except IOError as e:
            print(e)
        ssh.close()
        return (path[0], images,images)
    def groupDuplicate(self, dtpack):

        path, list = dtpack
        images = []
	
        #print(path)
        for img in list:
            print("Leng---------->>>>>> "+ str(len(img[1])))
            images.append(self.convert(img[1], ""))
        return (path[0], images,images)

    def loadBundleIntoCluster(self, path, offset=None, size=(-1)):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        images = []
        sortedpath = path[1]
        sortedpath.sort()
        try:
            for img_name in sortedpath:
                ftp = ssh.open_sftp()
                file = ftp.file(img_name, 'r', (-1))
                buf = file.read()
                imagbuf = imread(BytesIO(buf))
                if (imagbuf != None) and (hasattr(imagbuf, 'shape')):
                    images.append(imagbuf)
                else:
                    print("Warning: corrupted image skipped -->" + img_name)
                ftp.close()
        except IOError as e:
            print e
        ssh.close()
        return (path[0], images)

    def convert(self, img_object, params):
        # convert
        if(len(img_object.shape) >2):
            gray = cv2.cvtColor(img_object,cv2.COLOR_BGR2GRAY)
            return gray
        else:
            return img_object

    def estimate(self,img_object, params):
        knl_size, itns = params
        ret, thresh = cv2.threshold(img_object, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((knl_size, knl_size), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=itns)
        return opening

    def model(self, opening, params):
        knl_size, itns, dstnc, fg_ratio = params
        print(params)
        # sure background area
        kernel = np.ones((knl_size, knl_size), np.uint8)
        sure_bg = cv2.dilate(opening, kernel, iterations=itns)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, dstnc)
        ret, sure_fg = cv2.threshold(dist_transform, fg_ratio * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        print("Number of objects")
        print(ret)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Analysis
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        return markers

    def analysis(self, img_object, markers, params):
        markers = cv2.watershed(img_object, markers)
        img_object[markers == -1] = [255, 0, 0]
        return (img_object, markers)
    def commonTransform(self, params):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj=''
            try:
                procsd_obj = self.convert(imgaes, params)
            except Exception as e:
                print(e)

            return [(fname, imgaes, procsd_obj )]
        return common_transform

    def singleStepSegment(self, path, params):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        imagbuf = ''
        try:
            ftp = ssh.open_sftp()

            file = ftp.file(path, 'r', (-1))
            buf = file.read()
            imagbuf = imread(BytesIO(buf))

            ftp.close()
        except IOError as e:
            print(e)
        # ftp.close()
        ssh.close()
        if (imagbuf != None) and (hasattr(imagbuf, 'shape')):
            imgaes = imagbuf
            kernel_size, iterations, distance, forg_ratio = params
            procsd_obj = self.convert(imgaes, params)
            processed_obj = self.estimate(procsd_obj, ( kernel_size, iterations))
            model = self.model(processed_obj, params)
            processedimg, stats = self.analysis(imgaes, model, params)
        else:
            return (path, 0, 0)
        return (path, processedimg, stats)

    def singleHdfs(self, dpack, params):
        fname = dpack[0]
        # print(fname)
        imagbuf = imread(BytesIO(dpack[1]))
        if (imagbuf != None) and (hasattr(imagbuf, 'shape')):
            imgaes = imagbuf
            kernel_size, iterations, distance, forg_ratio = params
            procsd_obj = self.convert(imgaes, params)
            processed_obj = self.estimate(procsd_obj, ( kernel_size, iterations))
            model = self.model(processed_obj, params)
            processedimg, stats = self.analysis(imgaes, model, params)
        else:
            return [(fname, 0, 0)]
        return [(fname, processedimg, stats)]

    def commonEstimate(self, params):
        def common_estimate(datapack):
            fname, img, procsd_obj = datapack
            processed_obj = self.estimate(procsd_obj, params)
            return [(fname, img, processed_obj)]
        return common_estimate

    def commonModel(self,params):
        def common_model(datapack):
            fname,img, processed_obj = datapack
            model =self.model(processed_obj,params)
            return [(fname, img, model)]
        return common_model

    def commonAnalysisTransform(self, params):
        def common_transform(datapack):
            fname, img, model = datapack
            processedimg, stats = self.analysis(img, model, params)
            return [(fname, processedimg, stats)]
        return common_transform

    def extarct_feature_locally(self, feature_name, img):
        if feature_name in ["surf", "SURF"]:
            extractor = cv2.xfeatures2d.SURF_create()
        elif feature_name in ["sift", "SIFT"]:
            extractor = cv2.xfeatures2d.SIFT_create()
        elif feature_name in ["orb", "ORB"]:
            extractor = cv2.ORB_create()
        kp, descriptors = extractor.detectAndCompute(img_as_ubyte(img), None)
        return descriptors

    def estimate_feature(self, img, params):
        feature_name = params
        if feature_name in ["surf", "SURF"]:
            extractor = cv2.xfeatures2d.SURF_create()
        elif feature_name in ["sift", "SIFT"]:
            extractor = cv2.xfeatures2d.SIFT_create()
        elif feature_name in ["orb", "ORB"]:
            extractor = cv2.ORB_create()
        return extractor.detectAndCompute(img_as_ubyte(img), None)

    def saveResult(self, result):
        transport = paramiko.Transport((self.IMG_SERVER, 22))
        transport.connect(username=self.U_NAME, password=self.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        output = io.StringIO()
        try:
            sftp.stat(self.SAVING_PATH)
        except IOError, e:
            sftp.mkdir(self.SAVING_PATH)

        for lstr in result:
            output.write(lstr[0] + "\n")

        f = sftp.open(self.SAVING_PATH + "/"+str("result") + ".txt", 'wb')
        f.write(output.getvalue())

        sftp.close()


    def saveClusterResult(self,result):
         transport = paramiko.Transport((self.IMG_SERVER, 22))
         transport.connect(username=self.U_NAME, password=self.PASSWORD)
         sftp = paramiko.SFTPClient.from_transport(transport)
         # sftp.mkdir("/home/amit/A1/regResult/")
         dirs = set()
         dirs_list = []
         dirs_dict = dict()
         clusters = result
         try:
             sftp.stat(self.SAVING_PATH)
         except IOError, e:
             sftp.mkdir(self.SAVING_PATH)

         for lstr in clusters:
             group = lstr[0][1]
             img_name = lstr[0][0]
             if (group in dirs):
                 exists_list = dirs_dict[group]
                 exists_list.append(img_name)
                 dirs_dict.update({group: exists_list})
             else:
                 dirs_list = []
                 dirs_list.append(img_name)
                 dirs.add(group)
                 dirs_dict.update({group: dirs_list})

         for itms in dirs_dict.items():
             output = io.StringIO()
             for itm in itms[1]:
                 output.write(itm + "\n")

             f = sftp.open(self.SAVING_PATH + str(itms[0]) + ".txt", 'wb')

             f.write(output.getvalue())
         sftp.close()

    def common_write(self, result_path, sftp, fname, img, stat):
        try:
            sftp.stat(result_path)
        except IOError, e:
            sftp.mkdir(result_path)
        buffer = BytesIO()
        imsave(buffer, img, format='PNG')
        buffer.seek(0)
        dirs = fname.split('/')
        # print("write: " + fname)
        img_name = dirs[len(dirs) - 1]
        only_name = img_name.split('.')
        f = sftp.open(result_path + "/IMG_" + only_name[len(only_name)-2]+".png", 'wb')
        f.write(buffer.read())
        sftp.close()

    def all_write(self, result_path, sftp, fname, img, stat):
        try:
            sftp.stat(result_path)
        except IOError, e:
            sftp.mkdir(result_path)
        buffer = BytesIO()
        imsave(buffer, img, format='PNG')
        buffer.seek(0)
        dirs = fname.split('/')
        # print("write: " + fname)
        img_name = dirs[len(dirs) - 1]
        only_name = img_name.split('.')
        f = sftp.open(result_path + "/IMG_" + only_name[len(only_name)-2]+".png", 'wb')
        f.write(buffer.read())


    def commonSave(self, datapack):
        fname, procsd_img, stats = datapack
        transport = paramiko.Transport((self.IMG_SERVER, 22))
        transport.connect(username=self.U_NAME, password=self.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        self.common_write(self.SAVING_PATH, sftp, fname, procsd_img, stats)

    def save_img_bundle(self, data_pack):
        (fname, procsdimg, stats) = data_pack
        transport = paramiko.Transport((self.IMG_SERVER, 22))
        transport.connect(username=self.U_NAME, password=self.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        last_part = fname.split('/')[(len(fname.split('/')) - 1)]
        RESULT_PATH = (self.SAVING_PATH +"/"+ last_part)
        print("Resutl writing into :" + RESULT_PATH)
        try:
            sftp.stat(RESULT_PATH)
        except IOError as e:
            sftp.mkdir(RESULT_PATH)
        for (i, wrapped) in enumerate(procsdimg):
            buffer = BytesIO()
            imsave(buffer, wrapped, format='png')
            buffer.seek(0)
            f = sftp.open(((((RESULT_PATH + '/regi_') + last_part) + str(i)) + '.png'), 'wb')
            f.write(buffer.read())
        sftp.close()

    def saveLocalDir(self, data):
        filename, img = data
        print(filename)
        if (img != None) and (hasattr(img, 'shape')):
            cv2.imwrite(self.SAVING_PATH + "/" + filename.split('/')[(len(filename.split('/')) - 1)], img)


class ImageRegistration(ImgPipeline):
    def convert(self, img, params):

        imgaes = color.rgb2gray(img)
        return imgaes


    def convert_bundle(self, images, params):
        grey_imgs = []
        for img in images:
            try:
                grey_imgs.append(self.convert(img, params))
            except Exception as e:
                print e
        return grey_imgs

    def commonTransform(self, params):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj=[]
            try:
                procsd_obj = self.convert_bundle(imgaes, params)
            except Exception as e:
                print(e)

            return [(fname, imgaes, procsd_obj)]
        return common_transform

    def singleStep(self, path, params):

            no_of_match, ratio, reproj_thresh, indx = params
            procsd_obj=[]
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
            images = []
            sortedpath = path[1]
            sortedpath.sort()
            f_names = []
            try:
                for img_name in sortedpath:
                    ftp = ssh.open_sftp()
                    file = ftp.file(img_name, 'r', (-1))
                    buf = file.read()
                    imagbuf = imread(BytesIO(buf))
                    if (imagbuf != None) and (hasattr(imagbuf, 'shape')):
                        f_names.append(img_name)
                        images.append(imagbuf)
                    else:
                        print("Warning: corrupted image skipped -->" + img_name)
                    ftp.close()
            except IOError as e:
                print(e)
            ssh.close()
            (key_points, img_features) = self.bundle_estimate(images[0], params)
            img_points = np.float32([key_point.pt for key_point in key_points])
            wrappeds = []
            wrappeds.append(images[0])
            unsuccess = 0
            for img in images[1:]:

                (right_key_points, right_features) = self.bundle_estimate(img,params)
                right_key_points = np.float32([key_point.pt for key_point in right_key_points])

                if (right_features != None):
                    #imgs.append(img[imgind])
                    (back_proj_error, inlier_count, H) = self.match_and_tranform(right_key_points, right_features, img_points, img_features, no_of_match, ratio, reproj_thresh)
                    if ((H != None)):
                        wrapped = self.wrap_and_sample(H, img)
                        wrappeds.append(wrapped)
                        # Hs.append(H)
                        # back_proj_errors.append(back_proj_error)
                        # inlier_counts.append(inlier_count)
                    else:
                        print("Algorithm is not working properly")
                        unsuccess = unsuccess+1
                print("it is working:" + str(indx))



            return (f_names[0], wrappeds, unsuccess)

    def twoStep(self, dtpack, params):

            no_of_match, ratio, reproj_thresh, indx = params
            procsd_obj=[]
            f_name, images, objs = dtpack
            (key_points, img_features) = self.bundle_estimate(images[0], params)
            img_points = np.float32([key_point.pt for key_point in key_points])
            wrappeds = []
            wrappeds.append(images[0])
            unsuccess = 0
            for img in images[1:]:

                (right_key_points, right_features) = self.bundle_estimate(img,params)
                right_key_points = np.float32([key_point.pt for key_point in right_key_points])

                if (right_features != None):
                    #imgs.append(img[imgind])
                    (back_proj_error, inlier_count, H) = self.match_and_tranform(right_key_points, right_features, img_points, img_features, no_of_match, ratio, reproj_thresh)
                    if ((H != None)):
                        wrapped = self.wrap_and_sample(H, img)
                        wrappeds.append(wrapped)
                        # Hs.append(H)
                        # back_proj_errors.append(back_proj_error)
                        # inlier_counts.append(inlier_count)
                    else:
                        print("Algorithm is not working properly")
                        unsuccess = unsuccess+1
                print("it is working:" + str(indx))



            return (f_name, wrappeds, unsuccess)

    def bundle_estimate(self, img_obj, params):
        extractor = cv2.xfeatures2d.SIFT_create(nfeatures=100000)
        return extractor.detectAndCompute(img_as_ubyte(img_obj), None)


    def commonEstimate(self, params):
        def common_estimate(datapack):
            fname, imgs, procsd_obj = datapack
            img_key_points = []
            img_descriptors = []
            print("estimatinng for:" + fname + " " + str(len(imgs)))
            for img in procsd_obj:
                try:
                    (key_points, descriptors) = self.bundle_estimate(img,params)
                    key_points = np.float32([key_point.pt for key_point in key_points])
                except Exception as e:
                    descriptors = None
                    key_points = None
                img_key_points.append(key_points)
                img_descriptors.append(descriptors)
            procssd_entity = []
            print(str(len(img_descriptors)))
            procssd_entity.append(img_key_points)
            procssd_entity.append(img_descriptors)
            return [(fname, imgs, procssd_entity)]
        return common_estimate

    def match_and_tranform(self, keypoints_to_be_reg, features_to_be_reg, ref_keypoints, ref_features, no_of_match,ratio, reproj_thresh):
    #def match_and_tranform(self, features_to_be_reg, keypoints_to_be_reg, ref_features, ref_keypoints, no_of_match, ratio, reproj_thresh):
        matcher = cv2.DescriptorMatcher_create('BruteForce')
        raw_matches = matcher.knnMatch(features_to_be_reg, ref_features, 2)
        matches = [(m[0].trainIdx, m[0].queryIdx) for m in raw_matches if ((len(m) == 2) and (m[0].distance < (m[1].distance * ratio)))]
        back_proj_error = 0
        inlier_count = 0
        H =0
        if (len(matches) > no_of_match):
            src_pts = np.float32([keypoints_to_be_reg[i] for (_, i) in matches])
            dst_pts = np.float32([ref_keypoints[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
            src_t = np.transpose(src_pts)
            dst_t = np.transpose(dst_pts)

            for i in range(0, src_t.shape[1]):
                x_i = src_t[0][i]
                y_i = src_t[1][i]
                x_p = dst_t[0][i]
                y_p = dst_t[1][i]
                num1 = (((H[0][0] * x_i) + (H[0][1] * y_i)) + H[0][2])
                num2 = (((H[1][0] * x_i) + (H[1][1] * y_i)) + H[1][2])
                dnm = (((H[2][0] * x_i) + (H[2][1] * y_i)) + H[2][2])
                tmp = (((x_p - (num1 / (dnm ** 2))) + y_p) - (num2 / (dnm ** 2)))
                if (status[i] == 1):
                    back_proj_error += tmp
                    inlier_count += 1
        return (back_proj_error, inlier_count, H)
    def wrap_and_sample(self, transformed_img, ref_img):
        wrapped = cv2.warpPerspective(ref_img, transformed_img, (ref_img.shape[1], ref_img.shape[0]))
        return wrapped

    def commonModel(self,params):
        def common_model(datapack):
            fname,imgs, procssd_entity = datapack
            right_key_points = procssd_entity[0]
            img_features = procssd_entity[1]
            no_of_match, ratio, reproj_thresh, base_img_idx = params
            indx = 0
            Hs = []
            back_proj_errors = []
            inlier_counts = []
            #imgs = []
            print("Modeling for:"+str(len(imgs)))
            for imgind, right_features in enumerate(img_features):
                if (right_features is not None):
                    #imgs.append(img[imgind])
                    (back_proj_error, inlier_count, H) = self.match_and_tranform(right_key_points[imgind], right_features, right_key_points[base_img_idx], img_features[base_img_idx], no_of_match, ratio, reproj_thresh)
                    if ((H is not None)):
                        Hs.append(H)
                        back_proj_errors.append(back_proj_error)
                        inlier_counts.append(inlier_count)
                    else:
                        print("Algorithm is not working properly")
                print("it is working:" + str(imgind))
                indx = (indx + 1)
            Hs.insert(base_img_idx, np.identity(3))
            model = []
            model.append(Hs)
            model.append(back_proj_errors)
            model.append(inlier_counts)
            return [(fname, imgs, model)]
        return common_model

    def commonAnalysisTransform(self, params):
        def common_transform(datapack):
            (fname, imgs, model) = datapack
            H = model[0]
            wrappeds = []
            if(len(H) <1):
                print("H is empty algorithm is not working properly")
            for i, img in enumerate(imgs):
                wrapped = self.wrap_and_sample(H[i], img)
                wrappeds.append(wrapped)
            stats = []
            stats.append(H)
            stats.append(model[1])
            stats.append(model[2])
            return [(fname, wrappeds, stats)]
        return common_transform

    def write_register_images(self, data_pack):
        (fname, procsdimg, stats) = data_pack
        transport = paramiko.Transport((self.IMG_SERVER, 22))
        transport.connect(username=self.U_NAME, password=self.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        RESULT_PATH = (self.SAVING_PATH +"/"+ fname.split('/')[(len(fname.split('/')) - 1)])
        print("Resutl writing into :" + RESULT_PATH)
        try:
            sftp.stat(RESULT_PATH)
        except IOError as e:
            sftp.mkdir(RESULT_PATH)
        for (i, wrapped) in enumerate(procsdimg):
            buffer = BytesIO()
            imsave(buffer, wrapped, format='PNG')
            buffer.seek(0)
            f = sftp.open((((((RESULT_PATH + '/IMG_') + '0') + '_') + str(i)) + '.png'), 'wb')
            f.write(buffer.read())
        sftp.close()


class ImgMatching(ImgPipeline):

    def singleHdfs(self, dpack, params):
        #print(len(dpack))
        fname = dpack[0]
        #print(fname)
        imgbuf = imread(BytesIO(dpack[1]))
        feature2, algo = params
        procsd_obj = self.convert(imgbuf, (0))
        kp, descriptors = self.estimate_feature(procsd_obj, algo)

        try:
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # # print(feature2.value)
            # # Match descriptors.
            # matches = bf.match(procssd_entity, feature2.value)
            matcher = cv2.DescriptorMatcher_create('BruteForce')
            matches = matcher.knnMatch(descriptors, feature2.value, 2)
            if (matches is None or descriptors is None):
                print("error for file:" + fname)
                match = 0
            else:
                match = len(matches) / float(len(descriptors))
        except Exception as e:
            print(e)
            match = 0

        return [(fname, imgbuf, match)]

    def convert(self, img, params):
        imgaes = color.rgb2gray(img)
        return imgaes

    def commonTransform(self, params):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj = ''
            try:
                procsd_obj = self.convert(imgaes, params)
            except Exception as e:
                print(e)

            return [(fname, imgaes, procsd_obj)]

        return common_transform

    def commonEstimate(self, params):
        def common_estimate(datapack):

            fname, img, procsd_obj = datapack
            try:
                kp, descriptors = self.estimate_feature(procsd_obj,params)
            except Exception as e:
                descriptors = None
                print(e)
            return [(fname, img, descriptors)]

        return common_estimate


    def commonModel(self,params):
        def common_model(datapack):
            fname,img, procssd_entity = datapack
            feature2 = params
            try:
                # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # # print(feature2.value)
                # # Match descriptors.
                # matches = bf.match(procssd_entity, feature2.value)
                matcher = cv2.DescriptorMatcher_create('BruteForce')
                matches = matcher.knnMatch(procssd_entity, feature2.value, 2)
                if (matches is None or procssd_entity is None):
                    print("error for file:" + fname)
                    match = 0
                else:
                    match = len(matches) / float(len(procssd_entity))
            except Exception as e:
                print(e)
                match = 0

            return [(fname, img, match)]
        return common_model

class ImgCluster(ImgPipeline):

    def convert(self, img, params):

        imgaes = color.rgb2gray(img)
        return imgaes

    def commonTransform(self, params):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj = ''
            try:
                procsd_obj = self.convert(imgaes, params)
            except Exception as e:
                print(e)

            return [(fname, " ", procsd_obj)]

        return common_transform
    def featureExtract(self, path, params):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.IMG_SERVER, username=self.U_NAME, password=self.PASSWORD)
        imagbuf = ''
        try:
            ftp = ssh.open_sftp()

            file = ftp.file(path, 'r', (-1))
            buf = file.read()
            imagbuf = imread(BytesIO(buf))

            ftp.close()
        except IOError as e:
            print(e)
        # ftp.close()
        ssh.close()
        if (imagbuf != None) and (hasattr(imagbuf, 'shape')):
            imgaes = imagbuf
            procsd_obj = self.convert(imgaes, params)
            kp, descriptors = self.estimate_feature(procsd_obj, params)
            return (path, imgaes, descriptors)
        else:
            imgaes = None
            descriptors =None
            return (path, imgaes, descriptors)
    def singleHdfs(self, dpack, params):
        #print(len(dpack))
        fname = dpack[0]
        #print(fname)
        imgbuf = imread(BytesIO(dpack[1]))
        if (imgbuf != None) and (hasattr(imgbuf, 'shape')):
            imgaes = imgbuf
            procsd_obj = self.convert(imgaes, params)
            kp, descriptors = self.estimate_feature(procsd_obj, params)
            return [(fname, ' ', descriptors)]
        else:
            imgaes = None
            descriptors =None
            return [(fname, ' ', descriptors)]


    def commonEstimate(self, params):
        def common_estimate(datapack):

            fname, img, procsd_obj = datapack
            try:
                kp, descriptors = self.estimate_feature(procsd_obj,params)
            except Exception as e:
                descriptors = None
                print(e)
            return [(fname, " ", descriptors)]

        return common_estimate

    def commonModel(self, params):
        features, K, maxIterations = params
        model = KMeans.train(features, K, maxIterations, initializationMode="random")
        return model

    def commonAnalysisTransform(self, params):
        def common_transform(datapack):
            fname, img, procsdentity = datapack
            clusterCenters,pooling = params
            feature_matrix = np.array(procsdentity.tolist())
            clusterCenters = clusterCenters.value
            model = KMeansModel(clusterCenters)
            bow = np.zeros(len(clusterCenters))

            for x in feature_matrix:
                k = model.predict(x)
                dist = distance.euclidean(clusterCenters[k], x)
                if pooling == "max":
                    bow[k] = max(bow[k], dist)
                elif pooling == "sum":
                    bow[k] = bow[k] + dist
            clusters = bow.tolist()
            group = clusters.index(min(clusters)) + 1
            return [(fname, 'none', group)]

        return common_transform

    def saveClusterResult(self,result):
         transport = paramiko.Transport((self.IMG_SERVER, 22))
         transport.connect(username=self.U_NAME, password=self.PASSWORD)
         sftp = paramiko.SFTPClient.from_transport(transport)
         # sftp.mkdir("/home/amit/A1/regResult/")
         dirs = set()
         dirs_list = []
         dirs_dict = dict()
         clusters = result
         try:
             sftp.stat(self.SAVING_PATH)
         except IOError, e:
             sftp.mkdir(self.SAVING_PATH)

         for lstr in clusters:
             group = lstr[0][2]
             img_name = lstr[0][0]
             if (group in dirs):
                 exists_list = dirs_dict[group]
                 exists_list.append(img_name)
                 dirs_dict.update({group: exists_list})
             else:
                 dirs_list = []
                 dirs_list.append(img_name)
                 dirs.add(group)
                 dirs_dict.update({group: dirs_list})

         for itms in dirs_dict.items():
             output = io.StringIO()
             for itm in itms[1]:
                 output.write(itm + "\n")

             f = sftp.open(self.SAVING_PATH + "/"+str(itms[0]) + ".txt", 'wb')

             f.write(output.getvalue())
         sftp.close()

class FlowerCounter(ImgPipeline):
    common_size =(534, 800)
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    template_img = []
    avrg_histo_b = []
    def setTemplateandSize(self, img, size):
        self.common_size = size
        self.template_img=img

    def setRegionMatrix(self, mat):
        self.region_matrix = mat

    def setAvgHist(self, hist):
        self.avrg_histo_b = hist

    def convert(self, img_obj, params):
        img_asarray = np.array(img_obj)
        return img_asarray



    def getFlowerArea(self, img, hist_b_shifts, segm_B_lower, segm_out_value=0.99, segm_dist_from_zerocross=5):
        """
        Take the image given as parameter and highlight flowers applying a logistic function
        on the B channel. The formula applied is f(x) = 1/(1 + exp(K * (x - T))) being K and T constants
        calculated based on the given parameters.
        :param img: Image array
        :param fname: Image filename
        :param segm_out_value: Value of the logistic function output when the input is the lower B segmentation value i.e. f(S), where S = self.segm_B_lower + self.hist_b_shifts[fname]
        :param segm_dist_from_zerocross: Value that, when substracted from the lower B segmentation value, the output is 0.5 i.e. Value P where f(self.segm_B_lower + self.hist_b_shifts[fname] - P) = 0.5
        :return: Grayscale image highlighting flower pixels (pixels values between 0 and 1)
        """
        # Convert to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Get the B channel and convert to float
        img_B = np.array(img_lab[:, :, 2], dtype=np.float32)

        # Get the parameter T for the formula
        t_exp = segm_B_lower + hist_b_shifts - segm_dist_from_zerocross

        # Get the parameter K for the formula
        k_exp = np.log(1 / segm_out_value - 1) / segm_dist_from_zerocross

        # Apply logistic transformation
        img_B = 1 / (1 + np.exp(k_exp * (img_B - t_exp)))

        return img_B

    def estimate(self, img_object, params):
        plot_mask = params

        array_image = np.asarray(img_object)
        im_bgr = np.array(array_image)

        # Shift to grayscale
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        # Shift to LAB
        im_lab_plot = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)

        # Keep only plot pixels
        im_gray = im_gray[plot_mask > 0]
        im_lab_plot = im_lab_plot[plot_mask > 0]

        # Get histogram of grayscale image
        hist_G, _ = np.histogram(im_gray, 256, [0, 256])

        # Get histogram of B component
        hist_b, _ = np.histogram(im_lab_plot[:, 2], 256, [0, 256])
        histograms = []
        histograms.append(hist_b)
        histograms.append(hist_G)
        return histograms

    def model(self,processed_obj, params):
        hist_b = processed_obj[0]
        avg_hist_b = params
        # Calculate correlation
        correlation_b = np.correlate(hist_b, avg_hist_b, "full")

        # Get the shift on the X axis
        x_shift_b = correlation_b.argmax().astype(np.int8)
        return x_shift_b

    def analysis(self, img, model, params):
        flower_area_mask, segm_B_lower = params
        hist_b_shift = model
        # Get flower mask for this image

        # pil_image = PIL.Image.open(value).convert('RGB')
        open_cv_image = np.array(img)
        # print open_cv_image
        # img = open_cv_image[::-1].copy()
        img = open_cv_image[:, :, ::-1].copy()

        # Highlight flowers
        img_flowers = self.getFlowerArea(img, hist_b_shift, segm_B_lower, segm_dist_from_zerocross=8)

        # Apply flower area mask
        # print(img_flowers)
        # print(flower_area_mask)
        img_flowers[flower_area_mask == 0] = 0

        # Get number of flowers using blob counter on the B channel
        blobs = blob_doh(img_flowers, max_sigma=5, min_sigma=1)
        for bld in blobs:
            x, y, r = bld
            cv2.circle(img, (int(x), int(y)), int(r + 1), (0, 0, 0), 1)
        return (img, blobs)

    def crossProduct(self, p1, p2, p3):
        """
        Cross product implementation: (P2 - P1) X (P3 - P2)
        :param p1: Point #1
        :param p2: Point #2
        :param p3: Point #3
        :return: Cross product
        """
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        return v1[0] * v2[1] - v1[1] * v2[0]

    def userDefinePlot(self, img, bounds=None):
        """

        :param image: The image array that contains the crop
        :param bounds: Optionally user can set up previously the bounds without using GUI
         :return: The four points selected by user and the mask to apply to the image
        """
        # Initial assert
        if not isinstance(img, np.ndarray):
            print("Image is not a numpy array")
            return

        # Get image shape
        shape = img.shape[::-1]

        # Eliminate 3rd dimension if image is colored
        if len(shape) == 3:
            shape = shape[1:]

        # Function definitions
        def getMask(boundM):
            """
            Get mask from bounds
            :return: Mask in a numpy array
            """
            # Initialize mask
            # shapeM = img.shape[1::-1]
            mask = np.zeros(shape[::-1])

            # Get boundaries of the square containing our ROI
            minX = max([min([x[0] for x in boundM]), 0])
            minY = max([min([y[1] for y in boundM]), 0])
            maxX = min(max([x[0] for x in boundM]), shape[0])
            maxY = min(max([y[1] for y in boundM]), shape[1])

            # Reshape bounds
            # boundM = [(minX, minY), (maxX, minY), (minX, maxY), (maxX, maxY)]

            # Iterate through the containing-square and eliminate points
            # that are out of the ROI
            for x in range(minX, maxX):
                for y in range(minY, maxY):
                    h1 = self.crossProduct(boundM[2], boundM[0], (x, y))
                    h2 = self.crossProduct(boundM[3], boundM[1], (x, y))
                    v1 = self.crossProduct(boundM[0], boundM[1], (x, y))
                    v2 = self.crossProduct(boundM[2], boundM[3], (x, y))
                    if h1 > 0 and h2 < 0 and v1 > 0 and v2 < 0:
                        mask[y, x] = 255

            return mask

        # Check if bounds have been provided
        if isinstance(bounds, list):
            if len(bounds) != 4:
                print("Bounds length must be 4. Setting up GUI...")
            else:
                mask = getMask(bounds)
                return bounds, mask

        # Get image shape
        # shape = img.shape[1::-1]

        # Initialize boudaries
        height,width = self.common_size
        bounds = [(0, 0), ((height-1), 0), (0, (width-1)), ((height-1), (width-1))]

        # if plot == False:
        #    #for flower area
        #    bounds = [(308, 247), (923, 247), (308, 612), (923, 612)]


        # Get binary mask for the user-selected ROI
        mask = getMask(bounds)

        return bounds, mask

    # filenames = list_files("/data/mounted_hdfs_path/user/hduser/plot_images/2016-07-05_1207")


    def setPlotMask(self, bounds, imsize, mask=None):
        """
        Set mask of the plot under analysis
        :param mask: Mask of the plot
        :param bounds: Bounds of the plot
        """
        plot_bounds = None
        plot_mask = None
        # Initial assert
        if mask is not None:
            print(mask.shape)
            print(imsize)
            assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
            assert mask.shape == imsize, "Mask has a different size"
        assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

        # Store bounds
        plot_bounds = bounds

        # Store mask
        if mask is None:
            _, plot_mask = self.userDefinePlot(np.zeros(imsize), bounds)

        else:
            plot_mask = mask

        return plot_bounds, plot_mask

    def setFlowerAreaMask(self, region_matrix, mask, imsize):
        """
        Set mask of the flower area within the plot
        :param region_matrix = Region matrix representing the flower area
        :param mask: Mask of the flower area
        """

        # Initial assert
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Parameter 'mask' must be Numpy array"
            assert mask.shape == imsize, "Mask has a different size"

        # assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

        # Store bounds
        flower_region_matrix = region_matrix

        # Store mask
        flower_area_mask = mask

        return flower_area_mask

    def calculatePlotMask(self, images_bytes, imsize):
        """
        Compute plot mask
        """
        # Trace
        print("Computing plot mask...")

        # Read an image

        open_cv_image = np.array(images_bytes)

        # print open_cv_image
        # print (open_cv_image.shape)
        # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()


        p_bounds, p_mask = self.userDefinePlot(open_cv_image, None)

        # Store mask and bounds
        return self.setPlotMask(p_bounds, imsize, p_mask)

    def calculateFlowerAreaMask(self, region_matrix, plot_bounds, imsize):
        """
        Compute the flower area mask based on a matrix th        for bld in blob:
            x, y, r = bld
            cv2.circle(img, (int(x), int(y)), int(r + 1), (0, 0, 0), 1)at indicates which regions of the plot are part of the
        flower counting.
        :param region_matrix: Mmatrix reflecting which zones are within the flower area mask (e.g. in order to
        sample the center region, the matrix should be [[0,0,0],[0,1,0],[0,0,0]]
        """

        # Trace
        print("Computing flower area mask...")

        # Check for plot bounds
        assert len(plot_bounds) > 0, "Plot bounds not set. Please set plot bounds before setting flower area mask"

        # Convert to NumPy array if needed
        if not isinstance(region_matrix, np.ndarray):
            region_matrix = np.array(region_matrix)

        # Assert
        assert region_matrix.ndim == 2, 'region_matrix must be a 2D matrix'

        # Get the number of rows and columns in the region matrix
        rows, cols = region_matrix.shape

        # Get transformation matrix
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]]),
                                        np.float32(plot_bounds))

        # Initialize flower area mask
        fw_mask = np.zeros(imsize)

        # Go over the flower area mask and turn to 1 the marked areas in the region_matrix
        for x in range(cols):
            for y in range(rows):
                # Write a 1 if the correspondant element in the region matrix is 1
                if region_matrix[y, x] == 1:
                    # Get boundaries of this zone as a float32 NumPy array
                    bounds = np.float32([[x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]])
                    bounds = np.array([bounds])

                    # Transform points
                    bounds_T = cv2.perspectiveTransform(bounds, M)[0].astype(np.int)

                    # Get mask for this area
                    _, mask = self.userDefinePlot(fw_mask, list(bounds_T))

                    # Apply mask
                    fw_mask[mask > 0] = 255

        # Save flower area mask & bounds
        return self.setFlowerAreaMask(region_matrix, fw_mask, imsize)

    def computeAverageHistograms(self, hist_b_all):
        """
        Compute average B histogram
        """
        # Vertically stack all the B histograms
        avg_hist_B = np.vstack(tuple([h for h in hist_b_all]))

        # Sum all columns
        avg_hist_B = np.sum(avg_hist_B, axis=0)

        # Divide by the number of images and store
        avg_hist_b = np.divide(avg_hist_B, len(hist_b_all))

        return avg_hist_b

    def sumHistograms(self, hist_b_all):
        """
        Compute average B histogram
        """
        # Vertically stack all the B histograms

        avg_hist_B = np.vstack(tuple([h[0] for h in hist_b_all]))

        # Sum all columns
        avg_hist_B = np.sum(avg_hist_B, axis=0)
        avg_hist_b = np.divide(avg_hist_B, len(hist_b_all))

        return avg_hist_b


    def common_write(self, result_path, sftp, fname, img, stat):
        try:
            sftp.stat(result_path)
        except IOError, e:
            sftp.mkdir(result_path)
        buffer = BytesIO()
        imsave(buffer, img, format='PNG')
        buffer.seek(0)
        dirs = fname.split('/')
        print(fname)
        img_name = dirs[len(dirs) - 1]
        only_name = img_name.split('.')
        f = sftp.open(result_path + "/IMG_" + only_name[len(only_name)-2]+".png", 'wb')
        f.write(buffer.read())
        sftp.close()

class ImageStitching(ImgPipeline):
    def convert(self, img, params):
        imgaes = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return imgaes

    def convert_bundle(self, images, params):
        grey_imgs = []
        for img in images:
            try:
                grey_imgs.append(self.convert(img, params))
            except Exception as e:
                print(e)
        return grey_imgs

    def commonTransform(self, params):
        def common_transform(datapack):

            fname, images = datapack
            print("size: %", params)
            resize_imgs = []
            for img in images:
                resize_imgs.append(cv2.resize(img, params))
            procsd_obj = []
            try:
                procsd_obj = self.convert_bundle(resize_imgs, params)
            except Exception as e:
                print(e)

            return [(fname, resize_imgs, procsd_obj)]
        return common_transform

    def bundle_estimate(self, img_obj, params):
        extractor = cv2.xfeatures2d.SURF_create()
        return extractor.detectAndCompute(img_obj, None)


    def commonEstimate(self, params):
        def common_estimate(datapack):

            fname, imgs, procsd_obj = datapack
            img_key_points = []
            img_descriptors = []
            print("estimatinng for:" + fname + " " + str(len(imgs)))
            for img in procsd_obj:

                try:
                    (key_points, descriptors) = self.bundle_estimate(img,params)

                except Exception as e:
                    descriptors = None
                    key_points = None
                img_key_points.append(key_points)
                img_descriptors.append(descriptors)
            procssd_entity = []
            print(str(len(img_descriptors)))
            procssd_entity.append(img_key_points)
            procssd_entity.append(img_descriptors)
            return [(fname, imgs, procssd_entity)]
        return common_estimate

    def match(self, img_to_merge, feature, kp):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        kp_to_merge, feature_to_merge = self.getFeatureOfMergedImg(img_to_merge)
        matches = flann.knnMatch(feature,feature_to_merge,k=2)

        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = kp
            pointsPrevious = kp_to_merge

            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])

        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
        return H

    def match2(self, img_to_merge, b):
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        kp_to_merge, feature_to_merge = self.getFeatureOfMergedImg(img_to_merge)
        bkp, bf  = self.getFeatureOfMergedImg(b)
        matches = flann.knnMatch(bf, feature_to_merge, k=2)

        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = bkp
            pointsPrevious = kp_to_merge

            matchedPointsCurrent = np.float32([pointsCurrent[i].pt for (__, i) in good])
            matchedPointsPrev = np.float32([pointsPrevious[i].pt for (i, __) in good])

        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
        return H

    def getFeatureOfMergedImg(self, img):
        grey = self.convert(img,(0))
        return self.bundle_estimate(grey, (0))

    def commonModel(self,params):
        def common_model(datapack):

            fname,imgs, procssd_entity = datapack
            right_key_points = procssd_entity[0]
            img_features = procssd_entity[1]
            mid_indx = int(len(imgs)/2)
            start_indx = 1
            frst_img = imgs[0]
            print("shape of first img: % ", frst_img.shape)
            print("first phase {} ".format(mid_indx))
            for idx in range(mid_indx):
                H = self.match(frst_img, img_features[start_indx], right_key_points[start_indx])
                #H = self.match2(frst_img, imgs[start_indx])
                xh = np.linalg.inv(H)

                ds = np.dot(xh, np.array([frst_img.shape[1], frst_img.shape[0], 1]))
                ds = ds / ds[-1]
                print("final ds=>", ds)
                f1 = np.dot(xh, np.array([0, 0, 1]))
                f1 = f1 / f1[-1]
                xh[0][-1] += abs(f1[0])
                xh[1][-1] += abs(f1[1])
                ds = np.dot(xh, np.array([frst_img.shape[1], frst_img.shape[0], 1]))
                offsety = abs(int(f1[1]))
                offsetx = abs(int(f1[0]))
                dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
                print("image dsize =>", dsize) #(697, 373)
                tmp = cv2.warpPerspective(frst_img, xh, dsize)
                # cv2.imshow("warped", tmp)
                # cv2.waitKey()
                print("shape of img: %", imgs[start_indx].shape)
                print("shape of new {}".format(tmp.shape))

                tmp[offsety:imgs[start_indx].shape[0] + offsety, offsetx:imgs[start_indx].shape[1] + offsetx] = imgs[start_indx]
                frst_img = tmp
                start_indx = start_indx + 1
            model = []
            model.append(frst_img)
            model.append(procssd_entity)

            return [(fname, imgs, model)]
        return common_model

    # Homography is:  [[8.86033773e-01   6.59154846e-02   1.73593010e+02]
    #                  [-8.13825392e-02   9.77171622e-01 - 1.25890876e+01]
    # [-2.61821451e-04
    # 4.91986599e-05
    # 1.00000000e+00]]
    # Inverse
    # Homography: [[1.06785933e+00 - 6.26599825e-02 - 1.86161748e+02]
    #              [9.24787290e-02   1.01728698e+00 - 3.24694609e+00]
    # [2.75038650e-04 - 6.64548835e-05
    # 9.51418606e-01]]
    # final
    # ds = > [288.42753648  345.21227814    1.]
    # image
    # dsize = > (697, 373)
    # shape
    # of
    # new(373, 697, 3)

    def commonAnalysisTransform(self, params):
        def common_transform(datapack):

            fname, imgs, model = datapack
            procssd_entity = model[1]
            right_key_points = procssd_entity[0]
            img_features = procssd_entity[1]
            mid_indx = int(len(imgs) / 2)
            length = len(imgs)
            start_indx = mid_indx
            frst_img = model[0]
            print("second phase: %", start_indx)
            for idx in range(length-mid_indx):
                H = self.match(frst_img, img_features[start_indx], right_key_points[start_indx])

                txyz = np.dot(H, np.array([imgs[start_indx].shape[1], imgs[start_indx].shape[0], 1]))
                txyz = txyz / txyz[-1]
                dsize = (int(txyz[0]) + frst_img.shape[1], int(txyz[1]) + frst_img.shape[0])
                tmp = cv2.warpPerspective(imgs[start_indx], H, dsize)
                # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
                tmp = self.mix_and_match(frst_img, tmp)
                frst_img = tmp
                start_indx = start_indx + 1

            return [(fname, frst_img, '')]
        return common_transform


    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print leftImage[-1, -1]

        black_l = np.where(leftImage == np.array([0, 0, 0]))
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))
        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpedImage[j, i],
                                                                                                np.array([0, 0, 0]))):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if (np.array_equal(warpedImage[j, i], [0, 0, 0])):
                            # print "PIXEL"
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bw, gw, rw = warpedImage[j, i]
                                bl, gl, rl = leftImage[j, i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage

    def write_stitch_images(self, data_pack):
        (fname, procsdimg, stats) = data_pack
        transport = paramiko.Transport((self.IMG_SERVER, 22))
        transport.connect(username=self.U_NAME, password=self.PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        RESULT_PATH = (self.SAVING_PATH +"/"+ fname.split('/')[(len(fname.split('/')) - 1)])
        print("Resutl writing into :" + RESULT_PATH)
        try:
            sftp.stat(RESULT_PATH)
        except IOError as e:
            sftp.mkdir(RESULT_PATH)

        buffer = BytesIO()
        imsave(buffer, procsdimg, format='PNG')
        buffer.seek(0)
        f = sftp.open((((((RESULT_PATH + '/IMG_') + '0') + '_') + "stitched") + '.png'), 'wb')
        f.write(buffer.read())
        sftp.close()

class MosaicGenerator(ImgPipeline):
    def filter_matches(self, matches, ratio=0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])

        return filtered_matches

    def imageDistance(self, matches):

        sumDistance = 0.0

        for match in matches:
            sumDistance += match.distance

        return sumDistance

    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        (y, x) = image.shape[:2]

        base_p1[:2] = [0, 0]
        base_p2[:2] = [x, 0]
        base_p3[:2] = [0, y]
        base_p4[:2] = [x, y]

        max_x = None
        max_y = None
        min_x = None
        min_y = None

        for pt in [base_p1, base_p2, base_p3, base_p4]:

            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

            hp_arr = np.array(hp, np.float32)

            normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

            if (max_x == None or normal_pt[0, 0] > max_x):
                max_x = normal_pt[0, 0]

            if (max_y == None or normal_pt[1, 0] > max_y):
                max_y = normal_pt[1, 0]

            if (min_x == None or normal_pt[0, 0] < min_x):
                min_x = normal_pt[0, 0]

            if (min_y == None or normal_pt[1, 0] < min_y):
                min_y = normal_pt[1, 0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return (min_x, min_y, max_x, max_y)

    def convert(self, img, params):

        imgaes = cv2.GaussianBlur( color.rgb2gray(img), (5,5), 0)
        return imgaes
    def commonTransform(self, params):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj=''
            try:
                procsd_obj = self.convert(imgaes, params)
            except Exception as e:
                print(e)

            return [(fname, imgaes, procsd_obj )]
        return common_transform

    def estimate(self, img_obj, params):
        extractor = cv2.xfeatures2d.SIFT_create()
        img_key_points, img_descriptors =  extractor.detectAndCompute(img_as_ubyte(img_obj), None)
        return (img_key_points, img_descriptors)


    def commonEstimate(self, params):
        def common_estimate(datapack):
            fname, imgs, procsd_obj = datapack

            img_key_points, img_descriptors = self.estimate(procsd_obj, params)
            img_key_points = np.float32([key_point.pt for key_point in img_key_points])
            procssd_entity = []
            procssd_entity.append(img_key_points)
            procssd_entity.append(img_descriptors)


            return [(fname, imgs, procssd_entity)]
        return common_estimate

    def commonModel(self,params):
        def common_model(datapack):
            # print(datapack)
            fname,img, procssd_entity = datapack
            base_keypoints, base_descs, height, width = params
            next_keypoints = procssd_entity[0]
            next_descs = procssd_entity[1]
            inlier_ratio = 0
            model = []
            H = 0
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

            # print "\t Match Count: ", len(matches)

            matches_subset = self.filter_matches(matches)

            # print "\t Filtered Match Count: ", len(matches_subset)

            # distance = self.imageDistance(matches_subset)
            #
            # print "\t Distance from Key Image: ", distance
            #
            # averagePointDistance = distance / float(len(matches_subset))
            #
            # print "\t Average Distance: ", averagePointDistance
            # print "\t Filtered Match Count: ", len(matches_subset)
            kp1 = []
            kp2 = []
            left = width /10
            right = width - left
            top  = height /10
            bottom  = height - top
            outside_point_count  = 0
            print([left, right, top, bottom])
            if(len(matches_subset) > 4):
                for match in matches_subset:

                    bp1 = base_keypoints[match.trainIdx]
                    # print(bp1)
                    if(((bp1[1] <= top) or (bp1[1] >= bottom)) or ((bp1[0] <= left) or (bp1[0] >= right)) ):
                        outside_point_count = outside_point_count + 1

                    kp1.append(bp1)
                    bp2 = next_keypoints[match.queryIdx]
                    kp2.append(bp2)

                p1 = np.array([k for k in kp1])
                p2 = np.array([k for k in kp2])
                H = [p1,p2]
                # H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                # print("Points {} ".format(outside_point_count))
                # inlierRatio = float(np.sum(status)) / float(len(status))
                # outside_point_count = float(outside_point_count)/float(len(p1))
            else:
                inlierRatio = -1
                H = [0,0]
                outside_point_count = -1


            return [(outside_point_count, img, H, fname)]
        return common_model
    def tunedModel(self,params):
        def tuned_model(datapack):
            # print(datapack)
            fname, procssd_entity = datapack[1]
            base_keypoints, base_descs, height, width = params
            next_keypoints = procssd_entity[0]
            next_descs = procssd_entity[1]
            # base_keypoints = base_keypoints.value
            # base_descs = base_descs.value
            inlier_ratio = 0
            model = []
            H = 0
            matcher = cv2.DescriptorMatcher_create("BruteForce")
            matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

            # print "\t Match Count: ", len(matches)

            matches_subset = self.filter_matches(matches)

            # print "\t Filtered Match Count: ", len(matches_subset)

            # distance = self.imageDistance(matches_subset)
            #
            # print "\t Distance from Key Image: ", distance
            #
            # averagePointDistance = distance / float(len(matches_subset))
            #
            # print "\t Average Distance: ", averagePointDistance
            # print "\t Filtered Match Count: ", len(matches_subset)
            kp1 = []
            kp2 = []
            left = width /10
            right = width - left
            top  = height /10
            bottom  = height - top
            outside_point_count  = 0
            print([left, right, top, bottom])
            if(len(matches_subset) > 4):
                for match in matches_subset:

                    bp1 = base_keypoints[match.trainIdx]
                    # print(bp1)
                    if(((bp1[1] <= top) or (bp1[1] >= bottom)) or ((bp1[0] <= left) or (bp1[0] >= right)) ):
                        outside_point_count = outside_point_count + 1

                    kp1.append(bp1)
                    bp2 = next_keypoints[match.queryIdx]
                    kp2.append(bp2)

                p1 = np.array([k for k in kp1])
                p2 = np.array([k for k in kp2])
                H = [p1,p2]
                # H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                # print("Points {} ".format(outside_point_count))
                # inlierRatio = float(np.sum(status)) / float(len(status))
                # outside_point_count = float(outside_point_count)/float(len(p1))
            else:
                inlierRatio = -1
                H = [0,0]
                outside_point_count = -1


            return [(outside_point_count, H, datapack[0])]
        return tuned_model

    def efficient_model(self, datapack, params):
        # print(datapack)
        fname, procssd_entity = datapack[1]
        base_keypoints, base_descs, height, width = params
        next_keypoints = procssd_entity[0]
        next_descs = procssd_entity[1]
        # base_keypoints = base_keypoints.value
        # base_descs = base_descs.value
        inlier_ratio = 0
        model = []
        H = 0
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

        # print "\t Match Count: ", len(matches)

        matches_subset = self.filter_matches(matches)

        # print "\t Filtered Match Count: ", len(matches_subset)

        # distance = self.imageDistance(matches_subset)
        #
        # print "\t Distance from Key Image: ", distance
        #
        # averagePointDistance = distance / float(len(matches_subset))
        #
        # print "\t Average Distance: ", averagePointDistance
        # print "\t Filtered Match Count: ", len(matches_subset)
        kp1 = []
        kp2 = []
        left = width / 10
        right = width - left
        top = height / 10
        bottom = height - top
        outside_point_count = 0
        print([left, right, top, bottom])
        if (len(matches_subset) > 4):
            for match in matches_subset:

                bp1 = base_keypoints[match.trainIdx]
                # print(bp1)
                if (((bp1[1] <= top) or (bp1[1] >= bottom)) or ((bp1[0] <= left) or (bp1[0] >= right))):
                    outside_point_count = outside_point_count + 1

                kp1.append(bp1)
                bp2 = next_keypoints[match.queryIdx]
                kp2.append(bp2)

            p1 = np.array([k for k in kp1])
            p2 = np.array([k for k in kp2])
            H = [p1, p2]
            # H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
            # print("Points {} ".format(outside_point_count))
            # inlierRatio = float(np.sum(status)) / float(len(status))
            # outside_point_count = float(outside_point_count)/float(len(p1))
        else:
            inlierRatio = -1
            H = [0, 0]
            outside_point_count = -1

        return [(outside_point_count, H, datapack[0])]


class PlotSegment(ImgPipeline):
    def normalize_gaps(self, gaps, num_items):
        gaps = list(gaps)

        gaps_arr = np.array(gaps, dtype=np.float64)
        if gaps_arr.shape == (1,):
            gap_size = gaps_arr[0]
            gaps_arr = np.empty(num_items - 1)
            gaps_arr.fill(gap_size)
        elif gaps_arr.shape != (num_items - 1,):
            raise ValueError('gaps should have shape {}, but has shape {}.'
                             .format((num_items - 1,), gaps_arr.shape))
        return gaps_arr

    def get_repeated_seqs_2d_array(self, buffer_size, item_size, gaps, num_repeats_of_seq):
        start = buffer_size
        steps = gaps + item_size
        items = np.insert(np.cumsum(steps), 0, np.array(0)) + start
        return np.tile(items, (num_repeats_of_seq, 1))

    def set_plot_layout_relative_meters(self, buffer_blocwise_m, plot_width_m, gaps_blocs_m, num_plots_per_bloc, buffer_plotwise_m, plot_height_m, gaps_plots_m, num_blocs):

        # this one already has the correct grid shape.
        plot_top_left_corners_x = self.get_repeated_seqs_2d_array(buffer_blocwise_m, plot_width_m, gaps_blocs_m, num_plots_per_bloc)
        # this one needs to be transposed to assume the correct grid shape.
        plot_top_left_corners_y = self.get_repeated_seqs_2d_array(buffer_plotwise_m, plot_height_m, gaps_plots_m, num_blocs).T

        num_plots = num_blocs * num_plots_per_bloc
        plot_top_left_corners = np.stack((plot_top_left_corners_x, plot_top_left_corners_y)).T.reshape((num_plots, 2))

        plot_height_m_buffered = plot_height_m - 2 * buffer_plotwise_m
        plot_width_m_buffered = plot_width_m - 2 * buffer_blocwise_m

        plot_top_right_corners = np.copy(plot_top_left_corners)

        plot_top_right_corners[:, 0] = plot_top_right_corners[:, 0] + plot_width_m_buffered

        plot_bottom_left_corners = np.copy(plot_top_left_corners)
        plot_bottom_left_corners[:, 1] = plot_bottom_left_corners[:, 1] + plot_height_m_buffered

        plot_bottom_right_corners = np.copy(plot_top_left_corners)

        plot_bottom_right_corners[:, 0] = plot_bottom_right_corners[:, 0] + plot_width_m_buffered

        plot_bottom_right_corners[:, 1] = plot_bottom_right_corners[:, 1] + plot_height_m_buffered

        plots_all_box_coords = np.concatenate((plot_top_left_corners, plot_top_right_corners,
                                               plot_bottom_right_corners, plot_bottom_left_corners), axis=1)
        print(plots_all_box_coords)
        plots_corners_relative_m = plots_all_box_coords
        return plots_corners_relative_m

    def plot_segmentation(self, num_blocs, num_plots_per_bloc, plot_width, plot_height):
        # num_blocs = 5
        # num_plots_per_bloc = 17
        gaps_blocs = np.array([50])
        gaps_plots = np.array([5])
        buffer_blocwise = 1
        buffer_plotwise = 1
        # plot_width = 95
        # plot_height = 30


        num_blocs = int(num_blocs)
        num_plots_per_bloc = int(num_plots_per_bloc)
        buffer_blocwise_m = float(buffer_blocwise)
        buffer_plotwise_m = float(buffer_plotwise)
        plot_width_m = float(plot_width)
        plot_height_m = float(plot_height)

        if not all((num_blocs >= 1,
                    num_plots_per_bloc >= 1,
                    buffer_blocwise_m >= 0,
                    buffer_plotwise_m >= 0,
                    plot_width_m >= 0,
                    plot_height_m >= 0)):
            raise ValueError("invalid field layout parameters.")

        gaps_blocs_m = self.normalize_gaps(gaps_blocs, num_blocs)
        print(gaps_blocs_m)
        gaps_plots_m = self.normalize_gaps(gaps_plots, num_plots_per_bloc)
        print(gaps_plots_m)
        plots_corners_relative_m = None

        return self.set_plot_layout_relative_meters(buffer_blocwise_m, plot_width_m, gaps_blocs_m, num_plots_per_bloc, buffer_plotwise_m, plot_height_m, gaps_plots_m, num_blocs)

    def estimate(self,img_object, params):
        num_blocs, num_plots_per_bloc, p_width, p_height = params
        coord = self.plot_segmentation(num_blocs, num_plots_per_bloc, p_width, p_height)
        return coord

    def analysis(self, img, coord, params):

        xOffset, yOffset = params
        for i in range(coord.shape[0]):
            cv2.line(img, (int(coord[i, 0] + xOffset), int(coord[i, 1] + yOffset)),
                     (int(coord[i, 2] + xOffset), int(coord[i, 3] + yOffset)), (255, 255, 255), 2)
            cv2.line(img, (int(coord[i, 2] + xOffset), int(coord[i, 3] + yOffset)),
                     (int(coord[i, 4] + xOffset), int(coord[i, 5] + yOffset)), (255, 255, 255), 2)
            cv2.line(img, (int(coord[i, 4] + xOffset), int(coord[i, 5] + yOffset)),
                     (int(coord[i, 6] + xOffset), int(coord[i, 7] + yOffset)), (255, 255, 255), 2)
            cv2.line(img, (int(coord[i, 6] + xOffset), int(coord[i, 7] + yOffset)),
                     (int(coord[i, 0] + xOffset), int(coord[i, 1] + yOffset)), (255, 255, 255), 2)
        return (img, params)

    def commonEstimate(self, params):
        def common_estimate(datapack):
            fname, img = datapack
            b_lsist = params.value
            get_params = b_lsist[fname]
            extract_param = get_params.split()
            processed_obj = self.estimate(None, (int(extract_param[0]),int(extract_param[1]), int(extract_param[2]), int(extract_param[3]) ))
            return [(fname, img, processed_obj)]
        return common_estimate

    def commonAnalysisTransform(self, params):
        def common_transform(datapack):

            fname, img, model = datapack
            b_lsist = params.value
            get_params = b_lsist[fname]
            extract_param = get_params.split()
            processedimg, stats = self.analysis(img, model,(int(extract_param[4]), int(extract_param[5])))
            return [(fname, processedimg, stats)]
        return common_transform
#print(filenames)
#ftp = sc.broadcast(ftp)
def collectFiles(pipes, pattern):

    fil_list = pipes.collectFiles(pattern)
    return fil_list

def collectfromCSV(pipes, column):
    fil_list = pipes.collectImgFromCSV(column)
    return fil_list
def loadFiles(sc, pipes, fil_list):
    print("loading files............")
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    rdd = rdd.map(lambda kv: (pipes.loadIntoCluster(kv[1])))
    rdd = rdd.filter(lambda x: (x[1] !=None) and (hasattr(x[1], 'shape')) )
    return rdd

def test(sc, pipes, fil_list):
    print(fil_list)
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    rdd = rdd.map(lambda kv: (pipes.loadIntoCluster(kv[1])))
    rdd1 = rdd.filter(lambda kv: kv[0] == '/hadoopdata/t_one/IMG_0120_2 (2).tif')
    # print(rdd1.first())
    rdd = rdd.subtractByKey(rdd1)
    def rearrange(datapack):
        print(datapack[0])
        return (-1, datapack[0], datapack[1])
    rdd = rdd.map(rearrange)
    print(rdd.first())
    # print(rdd)
    # rdd = rdd.max(lambda x : x[0])
    # print(rdd[1])
    return rdd


def loadfromHDFS(hdfdir):
    rdd = sc.binaryFiles(hdfdir, npartitions)
    return rdd

def loadExecuteHDFS(sc, hdfdir, pipes):
    rdd = sc.binaryFiles(hdfdir, npartitions)
    rdd = rdd.flatMap(lambda kv: (pipes.splitHdfs(kv)))
    #rdd = rdd.filter(lambda x: (x[1] != None) and (hasattr(x[1], 'shape')))
    return rdd

def singleStepHDFS(sc, hdfdir, pipes, params):
    rdd = sc.binaryFiles(hdfdir, npartitions)
    rdd = rdd.flatMap(lambda kv: (pipes.singleHdfs(kv, params)))
    # rdd = rdd.filter(lambda x: (x[1] != None) and (hasattr(x[1], 'shape')))
    return rdd

def transform(pipes, rdd):
    print("transforming files............")
    rdd = rdd.flatMap(pipes.ParallelTransform)
    #print("Checking transform:" + str(rdd.count()))
    return rdd

def estimate(pipes, rdd, pattern):
    print("estimating files............")
    rdd = rdd.flatMap(pipes.parallel_feature_extract(pattern))

    return rdd

def common_transform(pipes, rdd, params):
    print("transforming files............{} ".format(params))
    rdd = rdd.flatMap(pipes.commonTransform(params))
    #print("Checking transform:" + str(rdd.count()))
    return rdd

def common_estimate(pipes, rdd, params):
    print("estimating files............")


    rdd = rdd.flatMap(pipes.commonEstimate(params))
    #print("Checking estimating:" + str(rdd.count()))
    return rdd

def common_model(pipes, rdd, params):
    print("modeling files............")
    # print(params)
    rdd = rdd.flatMap(pipes.commonModel(params))
    #print("Checking modeling:" + str(rdd.count()))
    return rdd
def tuned_model(pipes, rdd, params):
    print("modeling files............")
    # print(params)
    rdd = rdd.flatMap(pipes.tunedModel(params))
    #print("Checking modeling:" + str(rdd.count()))
    return rdd

def common_analysis(pipes, rdd, params):
    print("analysing files............")
    rdd = rdd.flatMap(pipes.commonAnalysisTransform(params))
    #print("Checking analysing:" + str(rdd.count()))common_result
    return rdd

def common_result(pipes, rdd):
    rdd.foreach(pipes.commonSave)

def result_local(pipes,rdd):
    all_list = collectAsArray(rdd)
    for data in all_list:
        pipes.saveLocalDir(data)

def cluster_model(pipes, rdd, params):
    K, itrns =params
    rdd = rdd.map(lambda x: (Row(fileName=x[0], features=x[2].tolist())))

    features = rdd.flatMap(lambda x: x['features']).cache()
    model = pipes.commonModel((features,K,itrns))
    return model

def cluster_analysis( pipes,rdd, params ):
    model  = params
    clusterCenters = model.clusterCenters
    clusterCenters = sc.broadcast(clusterCenters)
    features_bow = rdd.map(pipes.commonAnalysisTransform( (clusterCenters, 'sum')))
    return features_bow

def match_model(pipes, rdd, params):
    img_to_match, featurename = params
    key_pnts, feature2 = pipes.estimate_feature(pipes.convert(img_to_match,(0)),featurename )
    feature2 = sc.broadcast(feature2)
    print("matching run files............")
    rdd = rdd.flatMap(pipes.commonModel(feature2))
    return rdd

def match_analysis(pipes, rdd, params):
    factor  = params
    rdd = rdd.filter(lambda x: x[2] > factor)
    return rdd
def saveFileToHdfs(rdd, path):
    rdd.saveAsTextFile(path)

def collectResultAsName(pipes, rdd):
    pipes.saveResult(rdd.collect())

def collectBundle(pipes, pattern):
    image_sets_dirs = pipes.collectImagesSet(pattern)
    return image_sets_dirs

def loadBundle(sc, pipes, fil_list):
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    print("Loading data into cluster.......")
    rdd = rdd.map((lambda kv: pipes.loadBundleIntoCluster(kv[1])))
    #print("Load completed..." + str(rdd.count()))
    return rdd

def loadBundleSkipConversion(sc, pipes, fil_list):
    #global logger
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    print("Loading data into cluster.......")

    rdd = rdd.map((lambda kv: pipes.loadBundleIntoCluster_Skip_conversion(kv[1])))
    #logger.info("Data loading compoleted" + str(len(fil_list)))
    #print("Load completed..." + str(rdd.count()))
    return rdd
def localReadWrite(file_list, save_path):
    skipped = 0
    total = 0
    for path in file_list:
        img = cv2.imread(path)
        if(img != None) and (hasattr(img, 'shape')):
            cv2.imwrite(save_path+"/"+ path.split('/')[(len(path.split('/')) - 1)], img)
            total = total + 1
        else:
            skipped = skipped +1
        if(total>=10000):
           break;
    return skipped

def singleStepRegistration(sc, pipes, fil_list, params):
    #global logger
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    print("Loading data into cluster.......")

    rdd = rdd.map((lambda kv: pipes.singleStep(kv[1], params)))
    #logger.info("Data loading compoleted" + str(len(fil_list)))
    #print("Load completed..." + str(rdd.count()))
    return rdd

def twoStepRegistration(rdd, pipes, params):
    #global logger

    print("Loading data into cluster.......")

    rdd = rdd.map((lambda kv: pipes.twoStep(kv, params)))
    #logger.info("Data loading compoleted" + str(len(fil_list)))
    #print("Load completed..." + str(rdd.count()))
    return rdd

def singleStepSeg(sc, pipes, fil_list, params):
    #global logger
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    print("Loading data into cluster.......")

    rdd = rdd.map((lambda kv: pipes.singleStepSegment(kv[1], params)))
    #logger.info("Data loading compoleted" + str(len(fil_list)))
    #print("Load completed..." + str(rdd.count()))
    return rdd

def singleStepFeature(sc, pipes, fil_list, params):
    #global logger
    rdd = sc.parallelize(enumerate(fil_list), npartitions)
    print("Loading data into cluster.......")

    rdd = rdd.map((lambda kv: pipes.featureExtract(kv[1], params)))
    #logger.info("Data loading compoleted" + str(len(fil_list)))
    #print("Load completed..." + str(rdd.count()))
    return rdd

def collectAsArray(rdd):
    return rdd.collect()


def collectAndSave(rdd, pipes):
    list = rdd.collect()

    transport = paramiko.Transport((pipes.IMG_SERVER, 22))
    transport.connect(username=pipes.U_NAME, password=pipes.PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)
    for datapack in list:
        fname, procsd_img, stats = datapack
        pipes.all_write(pipes.SAVING_PATH, sftp, fname, procsd_img, stats)
    sftp.close()
def collectResultImages(pipes, rdd):
    rdd.foreach(pipes.write_register_images)

def collectImgBundle(pipes, rdd):
    rdd.foreach(pipes.save_img_bundle)

def collectStitchImages(pipes, rdd):
    rdd.foreach(pipes.write_stitch_images)

def createHdfsBundle(pipes, rdd):
    rdd = rdd.groupBy(lambda kv: pipes.matchpart(kv[0]))
    rdd = rdd.map((lambda kv: pipes.groupDuplicate(kv)))
    return rdd

def img_registration(sc, server, uname, password, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh, base_img_idx):
    print 'Executing from web................'
    pipes = ImageRegistration(server, uname, password)
    pipes.setLoadAndSavePath(data_path, save_path)
    file_bundles = collectBundle(pipes, img_type)
    rdd = loadBundle(sc, pipes, file_bundles)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd,(0))
    rdd = common_estimate(pipes, rdd, ('sift'))
    #rslt = collectAsArray(rdd)
    # for result in rslt:
    #     entt = result[2]
    #     print("chechking entity")
    #     for ent in entt:
    #         print(str(len(ent)))
    #         print(ent[0].shape)
    rdd = common_model(pipes, rdd, (no_of_match, ratio, reproj_thresh, base_img_idx))
    rdd = common_analysis(pipes, rdd, (0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    collectResultImages(pipes, rdd)

def img_registration2(sc, server, uname, password, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh, base_img_idx):
    print 'Executing from web................'
    pipes = ImageRegistration(server, uname, password)
    pipes.setLoadAndSavePath(data_path, save_path)
    file_bundles = pipes.collectImgsAsGroup(pipes.collectDirs(img_type))
    rdd = loadBundleSkipConversion(sc, pipes, file_bundles)
    processing_start_time = time()
    rdd = common_estimate(pipes, rdd, ('sift'))
    #rslt = collectAsArray(rdd)
    # for result in rslt:
    #     entt = result[2]
    #     print("chechking entity")
    #     for ent in entt:
    #         print(str(len(ent)))
    #         print(ent[0].shape)
    rdd = common_model(pipes, rdd, (no_of_match, ratio, reproj_thresh, base_img_idx))
    rdd = common_analysis(pipes, rdd, (0))
    collectImgBundle(pipes, rdd)
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def hdfsImgRegis():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    pipes = ImageRegistration(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    rdd = createHdfsBundle(pipes, rdd)
    rdd = common_estimate(pipes, rdd, ('sift'))
    rdd = common_model(pipes, rdd, (no_of_match, ratio, reproj_thresh, base_img_idx))
    rdd = common_analysis(pipes, rdd, (0))
    saveFileToHdfs(rdd, save_path)

def singleImgRegis():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    pipes = ImageRegistration(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    rdd = createHdfsBundle(pipes, rdd)
    rdd = twoStepRegistration(rdd, pipes, (no_of_match, ratio, reproj_thresh, base_img_idx))
    saveFileToHdfs(rdd, save_path)

def onestep_registration(sc, server, uname, password, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh, base_img_idx):
    print 'Executing from web................'
    pipes = ImageRegistration(server, uname, password)
    pipes.setLoadAndSavePath(data_path, save_path)
    file_bundles = pipes.collectImgsAsGroup(pipes.collectDirs(img_type))
    processing_start_time = time()
    rdd = singleStepRegistration(sc, pipes, file_bundles, (no_of_match, ratio, reproj_thresh, base_img_idx))
    collectImgBundle(pipes, rdd)
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def img_registration_fromCSV(sc, server, uname, password, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh, base_img_idx):
    print 'Executing from web................'
    pipes = ImageRegistration(server, uname, password)
    pipes.setCSVAndSavePath(data_path, save_path)
    file_bundles = pipes.collectImgsAsGroup(pipes.collectImgFromCSV("path"))
    rdd = loadBundleSkipConversion(sc, pipes, file_bundles)
    processing_start_time = time()
    rdd = common_estimate(pipes, rdd, ('sift'))
    #rslt = collectAsArray(rdd)
    # for result in rslt:
    #     entt = result[2]
    #     print("chechking entity")
    #     for ent in entt:
    #         print(str(len(ent)))
    #         print(ent[0].shape)
    rdd = common_model(pipes, rdd, (no_of_match, ratio, reproj_thresh, base_img_idx))
    rdd = common_analysis(pipes, rdd, (0))
    collectImgBundle(pipes, rdd)
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def img_matching(sc,server,uname,upass, data_path, save_path, img_type, img_to_search, ratio):
    pipes = ImgMatching(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (0))
    rdd = common_estimate(pipes, rdd, (0))
    rdd = match_model(pipes, rdd, (img_to_search, "orb"))
    rdd = match_analysis(pipes, rdd, (ratio))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    pipes.saveResult(rdd.collect())

def img_clustering(sc,server,uname,upass, data_path, save_path, img_type, K, iterations):
    pipes = ImgCluster(server, uname, upass)
    # pipes.setCSVAndSavePath(data_path, save_path)
    # files = collectfromCSV(pipes, "filename")
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()

    rdd = common_transform(pipes, rdd, (0))
    rdd = common_estimate(pipes, rdd, ("sift"))
    #print(rdd.count())
    model = cluster_model(pipes, rdd, (K,iterations))
    rdd = cluster_analysis(pipes, rdd, model)
    processing_end_time = time() - processing_start_time

    pipes.saveClusterResult(rdd.collect())
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))


def img_segmentation(sc,server,uname,upass, data_path, save_path, img_type,kernel_size, iterations, distance, forg_ratio):
    pipes = ImgPipeline(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = common_transform(pipes,rdd,(1))
    #rslt = collectAsArray(rdd)

    rdd = common_estimate(pipes, rdd, (kernel_size,iterations))
    # for result in rslt:
    #     entt = result[2]
    #     print("chechking entity")
    #     print(str(len(entt)))
    #     print(entt.shape)
    rdd = common_model(pipes,rdd, (kernel_size, iterations, distance, forg_ratio))
    rdd = common_analysis(pipes,rdd, (1))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    common_result(pipes,rdd)



#127.0.0.1 akm523 523@mitm /hadoopdata/reg_test_images /result '*' 3 2 0 .70
def callImgSeg():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        kernel_size = int(sys.argv[7])
        iterations = int(sys.argv[8])
        distance = int(sys.argv[9])
        fg_ratio = float(sys.argv[10])
        img_segmentation(sc, server, uname, upass, data_path, save_path, img_type, kernel_size, iterations, distance,
                         fg_ratio)

    except Exception as e:
        print(e)
        # img_matching(sc, server, uname, upass, data_path, save_path, img_type, img_to_seacrh, ratio = 0.55)
        # img_clustering(sc, server, uname, upass, csv_path, save_path, "'*'", K=3, iterations=20)
#fourfourframe001016.jpg
def hdfsMatching():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        img_to_search =str(sys.argv[7])
        ratio = float(sys.argv[8])
        pipes = ImgMatching(server, uname, upass)
        pipes.setLoadAndSavePath(data_path, save_path)
        processing_start_time = time()
        rdd = loadExecuteHDFS(sc, data_path, pipes)
        rdd = common_transform(pipes, rdd, (0))
        rdd = common_estimate(pipes, rdd, ("sift"))
        name, img_search = pipes.loadIntoCluster(img_to_search)
        rdd = match_model(pipes, rdd, (img_search, "sift"))
        rdd = match_analysis(pipes, rdd, (ratio))
        processing_end_time = time() - processing_start_time
        saveFileToHdfs(rdd, save_path)
        print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
        # pipes.saveResult(rdd.collect())

    except Exception as e:
        print(e)
        # img_matching(sc, server, uname, upass, data_path, save_path, img_type, img_to_seacrh, ratio = 0.55)
def singlehdfsMatching():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        img_to_search =str(sys.argv[7])
        ratio = float(sys.argv[8])
        pipes = ImgMatching(server, uname, upass)
        pipes.setLoadAndSavePath(data_path, save_path)
        processing_start_time = time()
        # rdd = common_transform(pipes, rdd, (0))
        # rdd = common_estimate(pipes, rdd, ("sift"))
        name, img_search = pipes.loadIntoCluster(img_to_search)
        conv_img = pipes.convert(img_search, (0))
        print("ok")
        points, feature=pipes.estimate_feature(conv_img,"sift")
        print("ok")
        feature = sc.broadcast(feature)
        rdd = singleStepHDFS(sc, data_path, pipes, (feature,"sift"))
        #rdd = match_model(pipes, rdd, (img_search, "sift"))
        rdd = match_analysis(pipes, rdd, (ratio))
        processing_end_time = time() - processing_start_time
        saveFileToHdfs(rdd, save_path)
        print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
        # pipes.saveResult(rdd.collect())

    except Exception as e:
        print(e)

def hdfsImgSeg():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        kernel_size = int(sys.argv[7])
        iterations = int(sys.argv[8])
        distance = int(sys.argv[9])
        fg_ratio = float(sys.argv[10])
        pipes = ImgPipeline(server, uname, upass)
        pipes.setLoadAndSavePath(data_path, save_path)
        #files = collectFiles(pipes, img_type)
        rdd = loadExecuteHDFS(sc, data_path, pipes)
        # print(rdd.count())
        processing_start_time = time()
        rdd = common_transform(pipes, rdd, (1))
        # rslt = collectAsArray(rdd)

        rdd = common_estimate(pipes, rdd, (kernel_size, iterations))
        # for result in rslt:
        #     entt = result[2]
        #     print("chechking entity")
        #     print(str(len(entt)))
        #     print(entt.shape)
        rdd = common_model(pipes, rdd, (kernel_size, iterations, distance, fg_ratio))
        rdd = common_analysis(pipes, rdd, (1))
        # collectAndSave(rdd,pipes)
        saveFileToHdfs(rdd, save_path)
        processing_end_time = time() - processing_start_time

        print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
        # common_result(pipes, rdd)

    except Exception as e:
        print(e)
        # img_matching(sc, server, uname, upass, data_path, save_path, img_type, img_to_seacrh, ratio = 0.55)
        # img_clustering(sc, server, uname, upass, csv_path, save_path, "'*'", K=3, iterations=20)

#/hadoop/img_data/10kcanola

def callClustering():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    k = int(sys.argv[7])
    iterations = int(sys.argv[8])
    img_clustering(sc, server, uname, upass, data_path, save_path, "'*'", k, iterations)

def hdfsClustering():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    k = int(sys.argv[7])
    iterations = int(sys.argv[8])
    pipes = ImgCluster(server, uname, upass)
    # pipes.setCSVAndSavePath(data_path, save_path)
    # files = collectfromCSV(pipes, "filename")
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    rdd = common_transform(pipes, rdd, (0))
    rdd = common_estimate(pipes, rdd, ("sift"))
    #print(rdd.count())
    model = cluster_model(pipes, rdd, (k, iterations))
    rdd = cluster_analysis(pipes, rdd, model)
    processing_end_time = time() - processing_start_time

    pipes.saveClusterResult(rdd.collect())
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def minhdfsClustering():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    k = int(sys.argv[7])
    iterations = int(sys.argv[8])
    pipes = ImgCluster(server, uname, upass)
    # pipes.setCSVAndSavePath(data_path, save_path)
    # files = collectfromCSV(pipes, "filename")
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = singleStepHDFS(sc, data_path, pipes, ("sift"))
    #print(rdd.count())
    model = cluster_model(pipes, rdd, (k, iterations))
    rdd = cluster_analysis(pipes, rdd, model)
    processing_end_time = time() - processing_start_time

    pipes.saveClusterResult(rdd.collect())
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def callMinClustering():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    k = int(sys.argv[7])
    iterations = int(sys.argv[8])
    pipes = ImgCluster(server, uname, upass)
    # pipes.setCSVAndSavePath(data_path, save_path)
    # files = collectfromCSV(pipes, "filename")
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)

    processing_start_time = time()
    rdd = singleStepFeature(sc, pipes, files, ("sift"))
    # print(rdd.count())
    model = cluster_model(pipes, rdd, (k, iterations))
    rdd = cluster_analysis(pipes, rdd, model)
    processing_end_time = time() - processing_start_time

    pipes.saveClusterResult(rdd.collect())
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))


def callOneStepSeg():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        kernel_size = int(sys.argv[7])
        iterations = int(sys.argv[8])
        distance = int(sys.argv[9])
        fg_ratio = float(sys.argv[10])
        pipes = ImgPipeline(server, uname, upass)
        pipes.setLoadAndSavePath(data_path, save_path)

        processing_start_time = time()
        files = collectFiles(pipes, img_type)
        rdd = singleStepSeg(sc, pipes, files, (kernel_size, iterations, distance, fg_ratio) )
        common_result(pipes, rdd)
        processing_end_time = time() - processing_start_time
        print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))


    except Exception as e:
        print(e)

def singleHdfsSeg():
    try:
        print(sys.argv[1:8])
        server = sys.argv[1]
        uname = sys.argv[2]
        upass = sys.argv[3]
        print('From web .............')
        print(uname)
        data_path = sys.argv[4]
        save_path = sys.argv[5]
        img_type = sys.argv[6]
        kernel_size = int(sys.argv[7])
        iterations = int(sys.argv[8])
        distance = int(sys.argv[9])
        fg_ratio = float(sys.argv[10])
        pipes = ImgPipeline(server, uname, upass)
        pipes.setLoadAndSavePath(data_path, save_path)

        processing_start_time = time()
        #files = collectFiles(pipes, img_type)
        rdd = singleStepHDFS(sc, data_path, pipes, (kernel_size, iterations, distance, fg_ratio) )
        saveFileToHdfs(rdd, save_path)
        processing_end_time = time() - processing_start_time
        print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))


    except Exception as e:
        print(e)
#127.0.0.1 akm523 523@mitm /hadoopdata/reg_test_images /result '*' 4 .75 0 0
def callImgReg():
    print sys.argv[1:10]
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    img_registration2(sc, server, uname, upass, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh,
                     base_img_idx)

def callOneStepReg():
    print sys.argv[1:10]
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    onestep_registration(sc, server, uname, upass, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh,
                     base_img_idx)

def callImgRegUsingSubFolders():
    print sys.argv[1:10]
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    img_registration(sc, server, uname, upass, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh,
                     base_img_idx)
# 127.0.0.1 akm523 523@mitm /hadoopdata/300ts/All/imglist.csv /result '*' 4 .75 0 0
def callImgRegCSV():
    print sys.argv[1:10]
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print 'From web .............'
    print uname
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    no_of_match = int(sys.argv[7])
    ratio = float(sys.argv[8])
    reproj_thresh = float(sys.argv[9])
    base_img_idx = int(sys.argv[10])
    img_registration_fromCSV(sc, server, uname, upass, data_path, save_path, img_type, no_of_match, ratio, reproj_thresh,
                     base_img_idx)
#127.0.0.1 uname pass /hadoopdata/flower /hadoopdata/flower_result '*' 155
def callFlowerCount():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    print(len(files))
    rdd = loadFiles(sc, pipes, files[0:10000])
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))

    hist_b_all = []
    all_array = collectAsArray(rdd)
    for i, element in enumerate(all_array):
        #print(element)
        histogrm = element[2]
        hist_b_all.append(histogrm[0])

    avg_hist_b = pipes.computeAverageHistograms(hist_b_all)  # Need to convert it in array
    rdd = common_model(pipes, rdd, (avg_hist_b))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     print(result)

    rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    common_result(pipes,rdd)

def hdfsFlowerCount():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files[0:10000])
    processing_start_time = time()
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))

    hist_b_all = []
    all_array = collectAsArray(rdd)
    for i, element in enumerate(all_array):
        #print(element)
        histogrm = element[2]
        hist_b_all.append(histogrm[0])

    avg_hist_b = pipes.computeAverageHistograms(hist_b_all)  # Need to convert it in array
    rdd = common_model(pipes, rdd, (avg_hist_b))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     print(result)

    rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    saveFileToHdfs(rdd, save_path)
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    #common_result(pipes,rdd)

def goodFlowerCount():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files[0:10000])
    processing_start_time = time()
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))
    sum_histogram = 0.0
    rdd.cache()
    length = rdd.count()
    part_length = int(length/npartitions)
    for i in range(npartitions):
        #start = i*part_length
        all = rdd.mapPartitionsWithIndex(lambda y, it: islice(it, 0, part_length) if y == i else []).collect()
        sum_histogram =  sum_histogram + pipes.sumHistograms(all)
    print(sum_histogram)
    # hist_b_all = []
    # all_array = collectAsArray(rdd)
    # for i, element in enumerate(all_array):
    #     #print(element)
    #     histogrm = element[2]
    #     hist_b_all.append(histogrm[0])
    #
    # avg_hist_b = pipes.computeAverageHistograms(hist_b_all)  # Need to convert it in array
    # rdd = common_model(pipes, rdd, (avg_hist_b))
    # # rslt = collectAsArray(rdd)
    # # for result in rslt:
    # #     print(result)
    #
    # rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    saveFileToHdfs(rdd, save_path)
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def goodHdfsCount():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    print(len(files))
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))

    sum_histogram = 0.0
    rdd.cache()
    length = rdd.count()
    temprdd =  rdd.flatMap(lambda kv: kv[2]).zipWithIndex().cache()
    part_length = int(length / npartitions)
    end = 0
    for i in range(npartitions):
        start = i*part_length
        end = start + part_length
        #all = rdd.mapPartitionsWithIndex(lambda y, it: islice(it, 0, part_length) if y == i else []).collect()
        all = temprdd.filter(lambda (key,index): index in range(start,end)).collect()
        sum_histogram = sum_histogram + pipes.sumHistograms(all)
    if(end < length):
        all = temprdd.filter(lambda (key, index): index in range(end,length )).collect()
        print(len(all))
        sum_histogram = sum_histogram + pipes.sumHistograms(all)
        sum_histogram = np.divide(sum_histogram, npartitions+1)
    else:
        sum_histogram = np.divide(sum_histogram, npartitions)
    print(sum_histogram)
    # hist_b_all = []
    # all_array = collectAsArray(rdd)
    # for i, element in enumerate(all_array):
    #     # print(element)
    #     histogrm = element[2]
    #     hist_b_all.append(histogrm[0])
    #
    # avg_hist_b = pipes.computeAverageHistograms(hist_b_all)  # Need to convert it in array
    # print(avg_hist_b)
    rdd = common_model(pipes, rdd, (sum_histogram))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     print(result)

    rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    common_result(pipes,rdd)

def singlehdfsFlower():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files[0:10000])
    processing_start_time = time()
    rdd = loadExecuteHDFS(sc, data_path, pipes)
    length = rdd.count()
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))

    sum_histogram = 0.0
    rdd.cache()

    temprdd =  rdd.flatMap(lambda kv: kv[2]).zipWithIndex().cache()
    part_length = int(length / npartitions)
    end = 0
    for i in range(npartitions):
        start = i*part_length
        end = start + part_length
        #all = rdd.mapPartitionsWithIndex(lambda y, it: islice(it, 0, part_length) if y == i else []).collect()
        all = temprdd.filter(lambda (key,index): index in range(start,end)).collect()
        sum_histogram = sum_histogram + pipes.sumHistograms(all)
    if(end < length):
        all = temprdd.filter(lambda (key, index): index in range(end,length )).collect()
        print(len(all))
        sum_histogram = sum_histogram + pipes.sumHistograms(all)
        sum_histogram = np.divide(sum_histogram, npartitions+1)
    else:
        sum_histogram = np.divide(sum_histogram, npartitions)
    # print(sum_histogram)
    # sum_histogram = broadcast(sum_histogram)
    rdd = common_model(pipes, rdd, (sum_histogram))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     print(result)

    rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    saveFileToHdfs(rdd, save_path)
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

#127.0.0.1 uname pass /hadoopdata/flower /hadoopdata/flower_result '*' 155
def callCSVFlowerCount():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    segm_B_lower = int(sys.argv[7])

    pipes = FlowerCounter(server, uname, upass)
    pipes.setCSVAndSavePath(data_path, save_path)
    files = pipes.collectImgFromCSV("path")
    print(len(files))
    rdd = loadFiles(sc, pipes, files[0:10000])
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))

    template = rdd.first()
    tem_img = template[2]
    height, width,channel = tem_img.shape
    pipes.setTemplateandSize(tem_img, (height, width))
    region_matrix = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pipes.setRegionMatrix(region_matrix)
    print(len(tem_img), tem_img.shape)
    plot_bound, plot_mask = pipes.calculatePlotMask(pipes.template_img, pipes.common_size)
    flower_mask = pipes.calculateFlowerAreaMask(pipes.region_matrix, plot_bound, pipes.common_size)

    rdd = common_estimate(pipes,rdd, (plot_mask))

    hist_b_all = []
    all_array = collectAsArray(rdd)
    for i, element in enumerate(all_array):
        #print(element)
        histogrm = element[2]
        hist_b_all.append(histogrm[0])

    avg_hist_b = pipes.computeAverageHistograms(hist_b_all)  # Need to convert it in array
    rdd = common_model(pipes, rdd, (avg_hist_b))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     print(result)

    rdd = common_analysis(pipes, rdd, (flower_mask, segm_B_lower))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    common_result(pipes,rdd)

# 127.0.0.1 akm523 523@mitm /hadoopdata/t_one /hadoopdata/stitched_result1 '*' "/hadoopdata/t_one/IMG_0120_2 (2).tif"

def generateMosaic():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    base_img_name = str(sys.argv[7])

    pipes = MosaicGenerator(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    print(len(files))
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))
    # print("Eelements in rdd: {}".format(rdd.count()))
    global npartitions
    # print(rdd.first())
    est_rdd = common_estimate(pipes,rdd, (1))
    est_rdd.cache()
    # print("Eelements in rdd: {}".format(est_rdd.count()))
    name_to_filter, base_img = pipes.loadIntoCluster(base_img_name)
    print("base image: {}".format(name_to_filter))
    conv_img = pipes.convert(base_img, (0))
    base_points, base_features = pipes.estimate(conv_img, (0))
    base_points = np.float32([key_point.pt for key_point in base_points])

    images_no = len(files)
    for i in range(images_no-1):
        if(name_to_filter != ''):
            # est_rdd = est_rdd.subtractByKey(name_to_filter)
            est_rdd = est_rdd.filter(lambda kv: kv[0] !=name_to_filter )
            # print(name_to_filter)

            rdd1 = est_rdd
            # est_rdd = est_rdd.flatMap()
            if(npartitions > ((images_no-1) - i)):
                est_rdd = est_rdd.repartition((images_no-1) - i)
            else:
                est_rdd = est_rdd.repartition(npartitions)
            # print("Eelements in rdd: {}".format(est_rdd.count()))


        height = base_img.shape[0]
        width = base_img.shape[1]
        print("height: {} width {}".format(height, width))
        # broadcast(base_points)
        points_rdd= common_model(pipes, rdd1, (base_points, base_features,height, width ))

        def printPack(datapack):

            print("Phase  {} {}".format(datapack[0], datapack[3]))

        # points_rdd.foreach(printPack)
        if(points_rdd.isEmpty()):
            break
        closestImage = points_rdd.max(lambda kv: kv[0])
        print("passed {} Imgname: {} ".format(closestImage[0], closestImage[3]))
        if(float(closestImage[0]) < 10):
            name_to_filter = closestImage[3]
            break
        p1, p2 = closestImage[2]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if (status == None ):
            name_to_filter = closestImage[3]
            break
        if (np.sum(status) < 4):
            name_to_filter = closestImage[3]
            break
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (closestImage[0] >= 10):  # and

            (min_x, min_y, max_x, max_y) = pipes.findDimensions(closestImage[1], H_inv) #converted image
            if(max_x > 10000 or max_y>10000):
                break

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            # print "Homography: \n", H
            # print "Inverse Homography: \n", H_inv
            # print "Min Points: ", (min_x, min_y)

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            print "New Dimensions: ", (img_w, img_h)

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(img_as_ubyte(base_img), move_h, (img_w, img_h))
            # print "Warped base image"

            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            next_img_warp = cv2.warpPerspective(closestImage[1], mod_inv_h, (img_w, img_h)) #This image should be main image
            # print "Warped next image"

            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w), np.uint8)

            # print "Enlarged Image Shape: ", enlarged_base_img.shape
            # print "Base Image Shape: ", base_img.shape
            # print "Base Image Warp Shape: ", base_img_warp.shape

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create a mask from the warped image for constructing masked composite
            (ret, data_map) = cv2.threshold(img_as_ubyte(color.rgb2gray(next_img_warp)), 0, 255, cv2.THRESH_BINARY)

            enlarged_base_img = cv2.add(enlarged_base_img, np.asarray(base_img_warp), mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)

            # Now add the warped image
            base_img = cv2.add(enlarged_base_img, img_as_ubyte(next_img_warp),
                                dtype=cv2.CV_8U)
            # pipes.commonSave((name_to_filter, base_img, 0))
            conv_img = pipes.convert(base_img, (0))
            base_points, base_features = pipes.estimate(conv_img, (0))
            base_points = np.float32([key_point.pt for key_point in base_points])
            name_to_filter = closestImage[3]
            print("To be filtered into next phase : {}".format(name_to_filter))
        else:
            name_to_filter = format(closestImage[3])
            print("Not matched : {}".format(closestImage[3]))
            break

    print("final image"+ name_to_filter)
    pipes.commonSave((name_to_filter, base_img, 0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    # common_result(pipes,rdd)


def hdfsMosaic():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    base_img_name = str(sys.argv[7])

    pipes = MosaicGenerator(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files)
    rdd = loadExecuteHDFS(sc, data_path,pipes)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))
    # print("Eelements in rdd: {}".format(rdd.count()))
    global npartitions
    # print(rdd.first())
    rdd = common_estimate(pipes,rdd, (1))
    #est_rdd.cache()
    # print("Eelements in rdd: {}".format(est_rdd.count()))

    # name_to_filter, base_img = pipes.loadIntoCluster(base_img_name)
    # print("base image: {}".format(name_to_filter))
    # conv_img = pipes.convert(base_img, (0))
    # base_points, base_features = pipes.estimate(conv_img, (0))
    # base_points = np.float32([key_point.pt for key_point in base_points])
    first_img = rdd.first()
    name_to_filter = first_img[0]
    base_img = first_img[1]
    base_points, base_features = first_img[2]

    temprdd = rdd.zipWithIndex().cache()
    images_no = temprdd.count()
    split_size = 80
    split = int(images_no/split_size)
    #base_img = np.zeros((100,100), np.float32)
    traversed_imgs = []
    traversed_imgs.append(name_to_filter)
    while len(traversed_imgs) < images_no:
        max_img = [-99,0,0,0]
        print("working")
        for i in range(split+1):
            start = i*split_size
            if(start-1 > images_no):
                break
            end = start + split_size
            if(end > images_no):
                end = images_no-1
            filtered_rdd = temprdd.filter(lambda (key, index): index in range(start, end+1) and key[0] not in traversed_imgs ).map(lambda (k, v): k)
            height = base_img.shape[0]
            width = base_img.shape[1]
            points_rdd = common_model(pipes, filtered_rdd, (base_points, base_features, height, width))
            closestImage = points_rdd.max(lambda kv: kv[0])
            if(closestImage[0] > max_img[0]):
                max_img = closestImage
        if (float(max_img[0]) < 10):
            name_to_filter = max_img[3]
            traversed_imgs.append(name_to_filter)
            continue
        p1,p2 = max_img[2]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if (status == None):
            name_to_filter = max_img[3]
            traversed_imgs.append(name_to_filter)
            continue
        if (np.sum(status) < 4):
            name_to_filter = max_img[3]
            traversed_imgs.append(name_to_filter)
            continue
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (max_img[0] >= 10):  # and

            (min_x, min_y, max_x, max_y) = pipes.findDimensions(max_img[1], H_inv)  # converted image
            if (max_x > 10000 or max_y > 10000):
                break

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            # print "Homography: \n", H
            # print "Inverse Homography: \n", H_inv
            # print "Min Points: ", (min_x, min_y)

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            print "New Dimensions: ", (img_w, img_h)

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(img_as_ubyte(base_img), move_h, (img_w, img_h))
            # print "Warped base image"

            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            next_img_warp = cv2.warpPerspective(max_img[1], mod_inv_h,
                                                (img_w, img_h))  # This image should be main image
            # print "Warped next image"

            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w,3), np.uint8)

            # print "Enlarged Image Shape: ", enlarged_base_img.shape
            # print "Base Image Shape: ", base_img.shape
            # print "Base Image Warp Shape: ", base_img_warp.shape

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create a mask from the warped image for constructing masked composite
            (ret, data_map) = cv2.threshold(img_as_ubyte(color.rgb2gray(next_img_warp)), 0, 255, cv2.THRESH_BINARY)
            print("matchimg length: {}".format(len(base_img_warp.shape)))
            print("Base length: {}".format(len(enlarged_base_img.shape)))
            enlarged_base_img = cv2.add(enlarged_base_img, np.asarray(base_img_warp), mask=np.bitwise_not(data_map),
                                        dtype=cv2.CV_8U)

            # Now add the warped image
            base_img = cv2.add(enlarged_base_img, img_as_ubyte(next_img_warp), dtype=cv2.CV_8U)
            # pipes.commonSave((name_to_filter, base_img, 0))
            conv_img = pipes.convert(base_img, (0))
            base_points, base_features = pipes.estimate(conv_img, (0))
            base_points = np.float32([key_point.pt for key_point in base_points])
            name_to_filter = max_img[3]
            traversed_imgs.append(name_to_filter)
            print("To be filtered into next phase : {}".format(name_to_filter))
        else:
            name_to_filter = format(max_img[3])
            traversed_imgs.append(name_to_filter)
            print("Not matched : {}".format(max_img[3]))
            continue



    print("final image list:"+ traversed_imgs)
    pipes.commonSave((name_to_filter, base_img, 0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    # common_result(pipes,rdd)

def goodMosaic():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    base_img_name = str(sys.argv[7])

    pipes = MosaicGenerator(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files)
    rdd = loadExecuteHDFS(sc, data_path,pipes)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))
    # print("Eelements in rdd: {}".format(rdd.count()))
    global npartitions
    # print(rdd.first())
    rdd = common_estimate(pipes,rdd, (1))
    #est_rdd.cache()
    # print("Eelements in rdd: {}".format(est_rdd.count()))

    # name_to_filter, base_img = pipes.loadIntoCluster(base_img_name)
    # print("base image: {}".format(name_to_filter))
    # conv_img = pipes.convert(base_img, (0))
    # base_points, base_features = pipes.estimate(conv_img, (0))
    # base_points = np.float32([key_point.pt for key_point in base_points])
    temprdd = rdd.zipWithIndex().map(lambda (key, ind): (ind, key)).cache()
    first_img = temprdd.lookup(0)
    print(first_img)
    name_to_filter = first_img[0][0]
    base_img = first_img[0][1]
    base_points, base_features = first_img[0][2]
    images_no = temprdd.count()
    split_size = 50
    split = int(images_no/split_size)
    #base_img = np.zeros((100,100), np.float32)
    traversed_imgs = []
    traversed_imgs.append(0)
    while len(traversed_imgs) < images_no:
        max_img = [-99,0,0]
        print("working")
        for i in range(split+1):
            start = i*split_size
            if(start-1 > images_no):
                break
            end = start + split_size
            if(end > images_no):
                end = images_no-1
            filtered_rdd = temprdd.filter(lambda (key, index): key in range(start, end+1) and key not in traversed_imgs ).map(lambda (k, v): (k,(v[0], v[2])))
            height = base_img.shape[0]
            width = base_img.shape[1]
            # base_points = sc.broadcast(base_points)
            # base_features = sc.broadcast(base_features)
            points_rdd = tuned_model(pipes, filtered_rdd, (base_points, base_features, height, width))
            if (points_rdd.isEmpty()):
                break
            closestImage = points_rdd.max(lambda kv: kv[0])
            if(closestImage[0] > max_img[0]):
                max_img = closestImage
        if (float(max_img[0]) < 10):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        p1,p2 = max_img[1]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if (status == None):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        if (np.sum(status) < 4):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (max_img[0] >= 10):  # and
            ext_element = temprdd.lookup(max_img[2])
            ext_img = ext_element[0][1]

            (min_x, min_y, max_x, max_y) = pipes.findDimensions(ext_img, H_inv)  # converted image
            if (max_x > 10000 or max_y > 10000):
                break

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            # print "Homography: \n", H
            # print "Inverse Homography: \n", H_inv
            # print "Min Points: ", (min_x, min_y)

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            print "New Dimensions: ", (img_w, img_h)

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(img_as_ubyte(base_img), move_h, (img_w, img_h))
            # print "Warped base image"

            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            next_img_warp = cv2.warpPerspective(ext_img, mod_inv_h,
                                                (img_w, img_h))  # This image should be main image
            # print "Warped next image"

            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w,3), np.uint8)

            # print "Enlarged Image Shape: ", enlarged_base_img.shape
            # print "Base Image Shape: ", base_img.shape
            # print "Base Image Warp Shape: ", base_img_warp.shape

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create a mask from the warped image for constructing masked composite
            (ret, data_map) = cv2.threshold(img_as_ubyte(color.rgb2gray(next_img_warp)), 0, 255, cv2.THRESH_BINARY)
            print("matchimg length: {}".format(len(base_img_warp.shape)))
            print("Base length: {}".format(len(enlarged_base_img.shape)))
            enlarged_base_img = cv2.add(enlarged_base_img, np.asarray(base_img_warp), mask=np.bitwise_not(data_map),
                                        dtype=cv2.CV_8U)

            # Now add the warped image
            base_img = cv2.add(enlarged_base_img, img_as_ubyte(next_img_warp), dtype=cv2.CV_8U)
            # pipes.commonSave((name_to_filter, base_img, 0))
            conv_img = pipes.convert(base_img, (0))
            base_points, base_features = pipes.estimate(conv_img, (0))
            base_points = np.float32([key_point.pt for key_point in base_points])
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            print("To be filtered into next phase : {}".format(name_to_filter))
        else:
            name_to_filter = format(max_img[2])
            traversed_imgs.append(name_to_filter)
            print("Not matched : {}".format(max_img[2]))
            continue



    print("final image list:"+ traversed_imgs)
    pipes.commonSave((name_to_filter, base_img, 0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    # common_result(pipes,rdd)

def efficientMosaic():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    base_img_name = str(sys.argv[7])

    pipes = MosaicGenerator(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files)
    rdd = loadExecuteHDFS(sc, data_path,pipes)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))
    # print("Eelements in rdd: {}".format(rdd.count()))
    global npartitions
    # print(rdd.first())
    rdd = common_estimate(pipes,rdd, (1))
    #est_rdd.cache()
    # print("Eelements in rdd: {}".format(est_rdd.count()))

    # name_to_filter, base_img = pipes.loadIntoCluster(base_img_name)
    # print("base image: {}".format(name_to_filter))
    # conv_img = pipes.convert(base_img, (0))
    # base_points, base_features = pipes.estimate(conv_img, (0))
    # base_points = np.float32([key_point.pt for key_point in base_points])
    temprdd = rdd.zipWithIndex().map(lambda (key, ind): (ind, key)).cache()
    filtered_rdd = temprdd.map(lambda (k, v): (k, (v[0], v[2])))
    first_img = temprdd.lookup(0)
    print(first_img)
    name_to_filter = first_img[0][0]
    base_img = first_img[0][1]
    base_points, base_features = first_img[0][2]
    images_no = temprdd.count()
    split_size = 100
    split = int(images_no/split_size)
    #base_img = np.zeros((100,100), np.float32)
    traversed_imgs = []
    traversed_imgs.append(0)
    to_reduce = 1
    while len(traversed_imgs) < images_no:
        max_img = [-99,0,0]
        print("working")
        height = base_img.shape[0]
        width = base_img.shape[1]
        filtered_rdd = filtered_rdd.filter(lambda (key, index): key not in traversed_imgs ).coalesce(images_no-to_reduce).cache()

        # base_points = sc.broadcast(base_points)
        # base_features = sc.broadcast(base_features)
        points_rdd = tuned_model(pipes, filtered_rdd, (base_points, base_features, height, width))
        to_reduce = to_reduce+1
        if (points_rdd.isEmpty()):
            break
        max_img = points_rdd.max(lambda kv: kv[0])
        # if(closestImage[0] > max_img[0]):
        #     max_img = closestImage

        if (float(max_img[0]) < 10):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        p1,p2 = max_img[1]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if (status == None):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        if (np.sum(status) < 4):
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            continue
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (max_img[0] >= 10):  # and
            ext_element = temprdd.lookup(max_img[2])
            ext_img = ext_element[0][1]

            (min_x, min_y, max_x, max_y) = pipes.findDimensions(ext_img, H_inv)  # converted image
            if (max_x > 10000 or max_y > 10000):
                break

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            # print "Homography: \n", H
            # print "Inverse Homography: \n", H_inv
            # print "Min Points: ", (min_x, min_y)

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            print "New Dimensions: ", (img_w, img_h)

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(img_as_ubyte(base_img), move_h, (img_w, img_h))
            # print "Warped base image"

            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            next_img_warp = cv2.warpPerspective(ext_img, mod_inv_h,
                                                (img_w, img_h))  # This image should be main image
            # print "Warped next image"

            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w,3), np.uint8)

            # print "Enlarged Image Shape: ", enlarged_base_img.shape
            # print "Base Image Shape: ", base_img.shape
            # print "Base Image Warp Shape: ", base_img_warp.shape

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create a mask from the warped image for constructing masked composite
            (ret, data_map) = cv2.threshold(img_as_ubyte(color.rgb2gray(next_img_warp)), 0, 255, cv2.THRESH_BINARY)
            print("matchimg length: {}".format(len(base_img_warp.shape)))
            print("Base length: {}".format(len(enlarged_base_img.shape)))
            enlarged_base_img = cv2.add(enlarged_base_img, np.asarray(base_img_warp), mask=np.bitwise_not(data_map),
                                        dtype=cv2.CV_8U)

            # Now add the warped image
            base_img = cv2.add(enlarged_base_img, img_as_ubyte(next_img_warp), dtype=cv2.CV_8U)
            # pipes.commonSave((name_to_filter, base_img, 0))
            conv_img = pipes.convert(base_img, (0))
            base_points, base_features = pipes.estimate(conv_img, (0))
            base_points = np.float32([key_point.pt for key_point in base_points])
            name_to_filter = max_img[2]
            traversed_imgs.append(name_to_filter)
            print("To be filtered into next phase : {}".format(name_to_filter))
        else:
            name_to_filter = format(max_img[2])
            traversed_imgs.append(name_to_filter)
            print("Not matched : {}".format(max_img[2]))
            continue



    print("final image list:"+ traversed_imgs)
    pipes.commonSave((name_to_filter, base_img, 0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    # common_result(pipes,rdd)



def generateCSVMosaic():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    base_img_name = str(sys.argv[7])

    pipes = MosaicGenerator(server, uname, upass)
    pipes.setCSVAndSavePath(data_path, save_path)
    files = pipes.collectImgFromCSV("path")
    print(len(files))
    rdd = loadFiles(sc, pipes, files)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, (1))
    # print("Eelements in rdd: {}".format(rdd.count()))
    global npartitions
    # print(rdd.first())
    est_rdd = common_estimate(pipes,rdd, (1))
    # print("Eelements in rdd: {}".format(est_rdd.count()))
    name_to_filter, base_img = pipes.loadIntoCluster(base_img_name)
    print("base image: {}".format(name_to_filter))
    conv_img = pipes.convert(base_img, (0))
    base_points, base_features = pipes.estimate(conv_img, (0))
    base_points = np.float32([key_point.pt for key_point in base_points])

    images_no = len(files)
    for i in range(images_no-1):
        if(name_to_filter != ''):
            # est_rdd = est_rdd.subtractByKey(name_to_filter)
            est_rdd = est_rdd.filter(lambda kv: kv[0] !=name_to_filter )
            # print(name_to_filter)

            rdd1 = est_rdd
            # est_rdd = est_rdd.flatMap()
            if(npartitions > ((images_no-1) - i)):
                est_rdd = est_rdd.repartition((images_no-1) - i)
            else:
                est_rdd = est_rdd.repartition(npartitions)
            # print("Eelements in rdd: {}".format(est_rdd.count()))


        height = base_img.shape[0]
        width = base_img.shape[1]
        print("height: {} width {}".format(height, width))
        # broadcast(base_points)
        points_rdd= common_model(pipes, rdd1, (base_points, base_features,height, width ))

        def printPack(datapack):

            print("Phase  {} {}".format(datapack[0], datapack[3]))

        # points_rdd.foreach(printPack)
        if(points_rdd.isEmpty()):
            break
        closestImage = points_rdd.max(lambda kv: kv[0])
        print("passed {} Imgname: {} ".format(closestImage[0], closestImage[3]))
        if(float(closestImage[0]) < 10):
            name_to_filter = closestImage[3]
            break
        p1, p2 = closestImage[2]
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        if (status == None ):
            name_to_filter = closestImage[3]
            break
        if (np.sum(status) < 4):
            name_to_filter = closestImage[3]
            break
        H = H / H[2, 2]
        H_inv = linalg.inv(H)

        if (closestImage[0] >= 10):  # and

            (min_x, min_y, max_x, max_y) = pipes.findDimensions(closestImage[1], H_inv) #converted image
            if(max_x > 10000 or max_y>10000):
                break

            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            # print "Homography: \n", H
            # print "Inverse Homography: \n", H_inv
            # print "Min Points: ", (min_x, min_y)

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            print "New Dimensions: ", (img_w, img_h)

            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(img_as_ubyte(base_img), move_h, (img_w, img_h))
            # print "Warped base image"

            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            next_img_warp = cv2.warpPerspective(closestImage[1], mod_inv_h, (img_w, img_h)) #This image should be main image
            # print "Warped next image"

            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()

            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w), np.uint8)

            # print "Enlarged Image Shape: ", enlarged_base_img.shape
            # print "Base Image Shape: ", base_img.shape
            # print "Base Image Warp Shape: ", base_img_warp.shape

            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp

            # Create a mask from the warped image for constructing masked composite
            (ret, data_map) = cv2.threshold(img_as_ubyte(color.rgb2gray(next_img_warp)), 0, 255, cv2.THRESH_BINARY)

            enlarged_base_img = cv2.add(enlarged_base_img, np.asarray(base_img_warp), mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)

            # Now add the warped image
            base_img = cv2.add(enlarged_base_img, img_as_ubyte(next_img_warp),
                                dtype=cv2.CV_8U)
            # pipes.commonSave((name_to_filter, base_img, 0))
            conv_img = pipes.convert(base_img, (0))
            base_points, base_features = pipes.estimate(conv_img, (0))
            base_points = np.float32([key_point.pt for key_point in base_points])
            name_to_filter = closestImage[3]
            print("To be filtered into next phase : {}".format(name_to_filter))
        else:
            name_to_filter = format(closestImage[3])
            print("Not matched : {}".format(closestImage[3]))
            break

    print("final image"+ name_to_filter)
    pipes.commonSave((name_to_filter, base_img, 0))
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
    # common_result(pipes,rdd)

def ImageIO():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]


    pipes = MosaicGenerator(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    files = collectFiles(pipes, img_type)
    skiped= localReadWrite(files, save_path)
    print(skiped)
    # rdd = loadFiles(sc, pipes, files)

#127.0.0.1 akm523 523@mitm /hadoopdata/stitching /hadoopdata/stitch_result '*' 480 320
def imageStitching():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    width = int(sys.argv[7])
    height = int(sys.argv[8])
    pipes = ImageStitching(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    file_bundles = collectBundle(pipes, img_type)
    rdd = loadBundle(sc, pipes, file_bundles)
    processing_start_time = time()
    rdd = common_transform(pipes, rdd, ((width, height)))
    rdd = common_estimate(pipes, rdd, (0))
    # rslt = collectAsArray(rdd)
    # for result in rslt:
    #     entt = result[2]
    #     print("chechking entity")
    #     for ent in entt[0]:
    #         print(str(len(ent)))
    #         print(ent)
    rdd = common_model(pipes, rdd, (0))
    rdd = common_analysis(pipes, rdd, (0))
    collectStitchImages(pipes, rdd)
    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

def plotSegment():
    print(sys.argv[1:8])
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    pipes = PlotSegment(server, uname, upass)
    pipes.setCSVAndSavePath(data_path, save_path)
    imgfile_withparams = pipes.ImgandParamFromCSV("path","param")
    rdd = loadFiles(sc, pipes, list(imgfile_withparams))
    imgfile_withparams = sc.broadcast(imgfile_withparams)
    rdd = common_estimate(pipes, rdd, (imgfile_withparams))
    rdd = common_analysis(pipes, rdd, (imgfile_withparams))
    common_result(pipes, rdd)

def testRdd():
    print(sys.argv[1:8])
    server = "127.0.0.1"
    uname = "akm523"
    upass = "523@mitm"
    print('From web .............')
    print(uname)
    data_path = "/hadoopdata/t_one"
    save_path = sys.argv[5]
    img_type = sys.argv[6]
    pipes = ImgPipeline(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    rdd = loadfromHDFS("hdfs://hpccdhm2.usask.ca:8020/user/akm523/testdata/10kcanola")
    rdd = rdd.take(1)
    print(rdd[0][0])
    imgbuf = imread(BytesIO(rdd[0][1]))
    cv2.imwrite("/hadoopdata/t_one/yaho.png", imgbuf)
    # files = collectFiles(pipes, img_type)
    # rdd = test(sc, pipes, files)
def hdfreadLocalwrite():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]


    pipes = ImgPipeline(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files)
    rdd = loadExecuteHDFS(sc, data_path,pipes)
    result_local(pipes, rdd)

def hdfreadwrite():
    server = sys.argv[1]
    uname = sys.argv[2]
    upass = sys.argv[3]
    print('From web .............')
    print(uname)
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    img_type = sys.argv[6]


    pipes = ImgPipeline(server, uname, upass)
    pipes.setLoadAndSavePath(data_path, save_path)
    # files = collectFiles(pipes, img_type)
    # print(len(files))
    # rdd = loadFiles(sc, pipes, files)
    rdd = loadExecuteHDFS(sc, data_path,pipes)
    saveFileToHdfs(rdd, save_path)

#phenodoop@sr-p2irc-big11:/hadoop/300_set
#akm523@onomi:/data/p2irc/Trials/Murayatrial/Murayatrial_1807201615m/Rededge/Images/All
#for mosaic: 127.0.0.1 akm523 523@mitm hdfs://hpccdhm2.usask.ca:8020/user/akm523/2mosaic /result '*'  155
if(__name__=="__main__"):
    spark = SparkSession.builder.appName("Plot registration from Server").getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    npartitions = 20
    #loger = sc._jvm.org.apache.log4j
    #logger = loger.LogManager.getRootLogger()
    #callOneStepReg()
    # callImgRegCSV()
    #callImgReg()
    #callImgSeg()
    #callFlowerCount()
    #callCSVFlowerCount()
    #imageStitching()
    #plotSegment()
    # testRdd()
    #generateMosaic()
    #generateCSVMosaic()
    #ImageIO()
    # callOneStepSeg()
    # callClustering()
    #callMinClustering()
    #testRdd()
    #hdfsImgSeg()
    #hdfsClustering()
    #hdfsMatching()
    #hdfsFlowerCount()
    #singlehdfsMatching()
    #minhdfsClustering()
    #singleHdfsSeg()
    # goodFlowerCount()
    # singlehdfsFlower()
    #hdfsMosaic()
    #goodMosaic()
    #efficientMosaic()
    #hdfreadLocalwrite()
    #hdfreadwrite()
    hdfsImgRegis()
    #singleImgRegis()
    sc.stop()
