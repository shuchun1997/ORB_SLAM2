/**
* This file is part of ORB-SLAM2.
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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

//这个函数的功能就是将strFile文件中的时间戳信息存到vTimestamps中去，将png图片路径存在vstrImageFilenames中去。
void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);  //函数定义

int main(int argc, char **argv)
{   //首先判断输入是否是4个参数，按照正确的输入应该是：./mono_tum path_to_vocabulary path_to_settings path_to_sequence。
    // 这里需要注意的是传入的参数分别对应argv[0],argv[1],argv[2],argv[3]。并且argv[0]对应的是程序本身
    if(argc != 4) //argc代表命令行参数的个数,当输入的参数个数不是4个的时候,执行报错提示
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;//程序异常结束
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;  //存图片?
    vector<double> vTimestamps;   //存图片时间
    string strFile = string(argv[3])+"/rgb.txt";  //用来存储rgb.txt这个文件
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();  //图片的个数

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //初始化SLAM系统
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;  //用来存放每张照片的追踪的时间
    vTimesTrack.resize(nImages);  //把容器的大小设置成容器数量的大小

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    //循环读取每一张图片，对图片中的特征点进行跟踪，记录跟踪的时间.
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);  //图片和时间戳传入slam系统

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());   //sort()函数用来排序,第一个参数排序起始地址,第二个是排序终止地址,第三个参数是排序类型,不行默认为从小到大
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;  //计算跟踪时间的中位数
    cout << "mean tracking time: " << totaltime/nImages << endl;    //跟踪时间的平均值

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)  //函数实现
{
    ifstream f;
    f.open(strFile.c_str());//c_str()为来保持和c语言兼容,把string转化成和char类型

    // skip first three lines
    string s0;
    getline(f,s0);  // getline(cin, inputLine)函数可以读取整行,包括空格. 其中cin是正在读取的输入流,而inputLine是接收输入字符串的string变量的名称
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())//判断是否读到文章末尾
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}
