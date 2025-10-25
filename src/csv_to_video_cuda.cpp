#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>

// 简单的CSV读取器
std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        data.push_back(row);
    }
    return data;
}

// 推断面部关键点数量
int inferFaceCount(const std::vector<std::vector<std::string>>& data) {
    int max_i = -1;
    if (data.empty()) return 0;
    for (const auto& col : data[0]) {
        if (col.find("face_") == 0 && col.find("_x") != std::string::npos) {
            size_t start = col.find('_') + 1;
            size_t end = col.find('_', start);
            try {
                int idx = std::stoi(col.substr(start, end - start));
                if (idx > max_i) max_i = idx;
            } catch (...) {}
        }
    }
    return max_i + 1;
}

// 加载关键点矩阵
std::vector<std::vector<cv::Point2f>> loadFaceKeypoints(const std::vector<std::vector<std::string>>& data, int faceCount) {
    std::vector<std::vector<cv::Point2f>> keypoints;
    size_t nFrames = data.size() - 1; // 跳过header
    keypoints.resize(nFrames, std::vector<cv::Point2f>(faceCount, cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN())));
    
    for (size_t frame = 0; frame < nFrames; ++frame) {
        for (int i = 0; i < faceCount; ++i) {
            std::string cx = "face_" + std::to_string(i) + "_x";
            std::string cy = "face_" + std::to_string(i) + "_y";
            auto itx = std::find(data[0].begin(), data[0].end(), cx);
            auto ity = std::find(data[0].begin(), data[0].end(), cy);
            if (itx != data[0].end() && ity != data[0].end()) {
                size_t idx = itx - data[0].begin();
                size_t idy = ity - data[0].begin();
                try {
                    float x = std::stof(data[frame + 1][idx]);
                    float y = std::stof(data[frame + 1][idy]);
                    keypoints[frame][i] = cv::Point2f(x, y);
                } catch (...) {}
            }
        }
    }
    return keypoints;
}

// 构建三角形（简化版，使用固定三角形或计算）
std::vector<cv::Vec3i> buildFaceTriangles(const std::vector<cv::Point2f>& points) {
    // 简化：假设68点面部模型，使用预定义三角形
    // 这里需要实际的Delaunay实现，但为了简单，使用固定三角形
    std::vector<cv::Vec3i> triangles;
    // 示例三角形（需要根据实际关键点调整）
    triangles.push_back(cv::Vec3i(0, 1, 2));
    // 添加更多...
    return triangles;
}

// CUDA加速的三角形变形
cv::cuda::GpuMat warpFaceTrianglesCUDA(const cv::cuda::GpuMat& src, const std::vector<cv::Point2f>& srcPts, const std::vector<cv::Point2f>& tgtPts, const std::vector<cv::Vec3i>& triangles) {
    cv::cuda::GpuMat result = src.clone();
    cv::cuda::GpuMat maskAccum(src.size(), CV_8U, cv::Scalar(0));
    
    for (const auto& tri : triangles) {
        std::vector<cv::Point2f> srcTri = {srcPts[tri[0]], srcPts[tri[1]], srcPts[tri[2]]};
        std::vector<cv::Point2f> tgtTri = {tgtPts[tri[0]], tgtPts[tri[1]], tgtPts[tri[2]]};
        
        // 检查NaN
        bool valid = true;
        for (auto& p : srcTri) if (std::isnan(p.x) || std::isnan(p.y)) valid = false;
        for (auto& p : tgtTri) if (std::isnan(p.x) || std::isnan(p.y)) valid = false;
        if (!valid) continue;
        
        cv::Rect bbox = cv::boundingRect(tgtTri);
        if (bbox.width <= 0 || bbox.height <= 0) continue;
        
        // 计算仿射变换
        cv::Mat M = cv::getAffineTransform(srcTri, tgtTri);
        cv::cuda::GpuMat MM;
        MM.upload(M);
        
        // 变形
        cv::cuda::GpuMat warped;
        cv::cuda::warpAffine(src, warped, MM, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        
        // 掩码
        cv::Mat mask(bbox.size(), CV_8U, cv::Scalar(0));
        std::vector<cv::Point> tgtTriInt;
        for (auto& p : tgtTri) tgtTriInt.push_back(cv::Point(p.x - bbox.x, p.y - bbox.y));
        cv::fillConvexPoly(mask, tgtTriInt, cv::Scalar(255));
        
        cv::cuda::GpuMat gpuMask;
        gpuMask.upload(mask);
        
        // 复制到结果
        cv::cuda::GpuMat roiSrc = warped(bbox);
        cv::cuda::GpuMat roiDst = result(bbox);
        roiSrc.copyTo(roiDst, gpuMask);
        
        // 更新累积掩码
        cv::cuda::GpuMat roiMaskAccum = maskAccum(bbox);
        cv::cuda::add(roiMaskAccum, gpuMask, roiMaskAccum);
    }
    
    return result;
}

// 填充缺失关键点
void fillMissingKeypoints(std::vector<std::vector<cv::Point2f>>& keypoints) {
    // 简化：用前一帧或后一帧填充
    for (size_t frame = 0; frame < keypoints.size(); ++frame) {
        for (size_t kp = 0; kp < keypoints[frame].size(); ++kp) {
            if (std::isnan(keypoints[frame][kp].x)) {
                // 查找最近的有效帧
                cv::Point2f val;
                for (int offset = -1; offset <= 1; offset += 2) {
                    int f = frame + offset;
                    if (f >= 0 && f < keypoints.size() && !std::isnan(keypoints[f][kp].x)) {
                        val = keypoints[f][kp];
                        break;
                    }
                }
                keypoints[frame][kp] = val;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <source_image> <csv_file> <output_dir>" << std::endl;
        return -1;
    }
    
    std::string sourceImg = argv[1];
    std::string csvFile = argv[2];
    std::string outputDir = argv[3];
    
    cv::Mat srcImg = cv::imread(sourceImg);
    if (srcImg.empty()) {
        std::cerr << "Cannot read source image: " << sourceImg << std::endl;
        return -1;
    }
    
    auto csvData = readCSV(csvFile);
    int faceCount = inferFaceCount(csvData);
    auto keypoints = loadFaceKeypoints(csvData, faceCount);
    fillMissingKeypoints(keypoints);
    
    if (keypoints.empty()) {
        std::cerr << "No keypoints loaded" << std::endl;
        return -1;
    }
    
    std::vector<cv::Point2f> srcKps = keypoints[0];
    auto triangles = buildFaceTriangles(srcKps);
    
    // 创建输出目录
    std::string frameDir = outputDir + "/generated_frames";
    cv::utils::fs::createDirectories(frameDir);
    
    cv::cuda::GpuMat gpuSrc;
    gpuSrc.upload(srcImg);
    
    for (size_t idx = 0; idx < keypoints.size(); ++idx) {
        cv::cuda::GpuMat warped;
        if (idx == 0) {
            warped = gpuSrc.clone();
        } else {
            warped = warpFaceTrianglesCUDA(gpuSrc, srcKps, keypoints[idx], triangles);
        }
        
        cv::Mat cpuWarped;
        warped.download(cpuWarped);
        
        char filename[256];
        sprintf(filename, "%s/frame_%06d.png", frameDir.c_str(), (int)idx);
        cv::imwrite(filename, cpuWarped);
        
        std::cout << "Generated frame " << idx << std::endl;
    }
    
    std::cout << "Done!" << std::endl;
    return 0;
}