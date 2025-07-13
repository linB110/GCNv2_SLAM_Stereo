/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "GCNextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


void nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.rows; i++){

        int u = (int) det.at<float>(i, 0);
        int v = (int) det.at<float>(i, 1);
        // float conf = det.at<float>(i, 2);

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    grid.setTo(0);
    inds.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;
    }
    
    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x + dist_thresh;
        int vv = (int) pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                grid.at<char>(vv + k, uu + j) = 0;
                
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++){
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                pts.push_back(cv::KeyPoint(pts_raw[select_ind].x * ratio_width, pts_raw[select_ind].y * ratio_height, 1.0f));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }
    
    descriptors.create(select_indice.size(), 32, CV_8U);

    for (int i=0; i<select_indice.size(); i++)
    {
        for (int j=0; j<32; j++)
        {
            descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
        }
    }
}



GCNextractor::GCNextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         int _iniThFAST, int _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    const char *net_fn = getenv("GCN_PATH");
    net_fn = (net_fn == nullptr) ? "gcn2.pt" : net_fn;
    try {
        module = std::make_shared<torch::jit::Module>(torch::jit::load(net_fn));
        //std::cout << "[GCNextractor] Loaded GCN model from: " << net_fn << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[GCNextractor] ERROR: Unable to load model from: " << net_fn << std::endl;
        std::cerr << e.what() << std::endl;
        exit(-1);
    }
}

void GCNextractor::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors)
{
    //std::cout << "[GCNextractor] operator() called" << std::endl;

    if (_image.empty()) {
        std::cerr << "[GCNextractor] Input image is empty!" << std::endl;
        return;
    }

    cv::Mat image = _image.getMat();
    //std::cout << "[GCNextractor] Original image size: " << image.cols << "x" << image.rows << ", type: " << image.type() << std::endl;

    if (image.type() != CV_8UC1) {
        std::cerr << "[GCNextractor] ERROR: Input must be grayscale (CV_8UC1)!" << std::endl;
        return;
    }

    // Store original image pyramid for later use (e.g., stereo matching)
    mvImagePyramid[0] = image.clone();
    for (int level = 1; level < nlevels; ++level) {
        float scale = 1.0f / mvScaleFactor[level];
        cv::resize(mvImagePyramid[level - 1], mvImagePyramid[level],
                   cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }

    // Resize to network input
    int net_w = 320, net_h = 240;
    int border = 8, dist_thresh = 4;
    if (getenv("FULL_RESOLUTION")) {
        net_w = 640;
        net_h = 480;
        border = 16;
        dist_thresh = 8;
    }

    float ratio_width = float(image.cols) / net_w;
    float ratio_height = float(image.rows) / net_h;

    cv::Mat img;
    image.convertTo(img, CV_32FC1, 1.0f / 255.0f);
    cv::resize(img, img, cv::Size(net_w, net_h));

    //std::cout << "[GCNextractor] Resized image to: " << img.cols << "x" << img.rows << std::endl;

    torch::Device device(torch::kCUDA);
    //std::cout << "[GCNextractor] Using device: CUDA" << std::endl;

    at::Tensor img_tensor;
    try {
        img_tensor = torch::from_blob(img.data, {1, 1, img.rows, img.cols}, torch::kFloat32).clone().to(device);
    } catch (const std::exception& e) {
        std::cerr << "[GCNextractor] ERROR: Tensor copy failed: " << e.what() << std::endl;
        return;
    }
    
    if (!img_tensor.defined() || img_tensor.numel() == 0) {
        std::cerr << "[GCNextractor] ERROR: img_tensor is undefined or empty!" << std::endl;
        return;
    }

    torch::IValue output_ivalue;
    try {
        output_ivalue = module->forward({img_tensor});
    } catch (const c10::Error& e) {
        std::cerr << "[GCNextractor] GCN forward() error: " << e.what() << std::endl;
        return;
    }

    if (!output_ivalue.isTuple()) {
        std::cerr << "[GCNextractor] ERROR: GCN output is not a tuple!" << std::endl;
        return;
    }

    auto output_tuple = output_ivalue.toTuple();
    if (output_tuple->elements().size() < 2) {
        std::cerr << "[GCNextractor] ERROR: Output tuple has fewer than 2 elements!" << std::endl;
        return;
    }

    at::Tensor pts_tensor, desc_tensor;
    try {
        pts_tensor = output_tuple->elements()[0].toTensor().to(torch::kCPU).squeeze();
        desc_tensor = output_tuple->elements()[1].toTensor().to(torch::kCPU).squeeze();
    } catch (const std::exception& e) {
        std::cerr << "[GCNextractor] ERROR: Tensor extraction failed: " << e.what() << std::endl;
        return;
    }

    //std::cout << "[GCNextractor] Tensor shapes - pts: " << pts_tensor.sizes() << ", desc: " << desc_tensor.sizes() << std::endl;

    if (pts_tensor.numel() == 0 || desc_tensor.numel() == 0) {
        std::cerr << "[GCNextractor] ERROR: Output tensors are empty!" << std::endl;
        return;
    }

    cv::Mat pts_mat(pts_tensor.size(0), 3, CV_32FC1, pts_tensor.data_ptr<float>());
    cv::Mat desc_mat(desc_tensor.size(0), 32, CV_8UC1, desc_tensor.data_ptr<uint8_t>());

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    //std::cout << "[GCNextractor] Running NMS..." << std::endl;

    nms(pts_mat, desc_mat, keypoints, descriptors,
        border, dist_thresh, net_w, net_h, ratio_width, ratio_height);

    //std::cout << "[GCNextractor] Keypoints after NMS: " << keypoints.size() << std::endl;
    //std::cout << "[GCNextractor] Descriptor size: " << descriptors.rows << "x" << descriptors.cols << std::endl;

    _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());

    if (!keypoints.empty()) {
        _descriptors.create((int)keypoints.size(), 32, CV_8U);
        descriptors.copyTo(_descriptors.getMat());
    } else {
        _descriptors.release();
    }
}



} //namespace ORB_SLAM
