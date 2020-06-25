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


#include    "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>

#include <unistd.h>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    //tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //,如果是双目,tracking过程都会用到mpORBextractorRight作为右目特征点提取器
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); //特征点检测,这个是单目初始化的时候用的特征提取,比mpORBextractorLeft作为特征点提取器提取的特征点个数多一倍

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

//tracking线程在获取图像数据后，会传给函数GrabImageStereo、GrabImageRGBD或GrabImageMonocular进行预处理,构造关键帧
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;//读取图像

    //将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    //构造Frame
    //构建成当前帧 并对当前帧进行ID标号,对图像提取ORB特征点,做左右图像的特征点匹配,计算视差,深度,得到mappoints等
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET) // 没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();//跟踪,调用Track()函数

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    // track包含两部分：估计运动、跟踪局部地图
    // mState为tracking的状态
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET)//表示当前没有图片。处于NO_IMAGES_YET，当新的一帧来临时，将线程状态改变为NOT_INITIALIZED。
    {
        mState = NOT_INITIALIZED;//表示当前没有初始化追踪线程。处于NOT_INITIALIZED，则针对单目相机和双目相机/RGBD相机进行不同的初始化。
    }

    mLastProcessedState=mState; //mLastProcessedState存储了Tracking最新的状态，用于FrameDrawer中的绘制

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // 初始化
    if(mState==NOT_INITIALIZED)//判断是否初始化
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();//双目和rgbd地图的初始化
        else
            MonocularInitialization();//判断是否初始化,重置参考帧

        mpFrameDrawer->Update(this); //更新帧的观测器

        if(mState!=OK)//表示当前追踪线程丢失——注意这里的线程状态都是指当前帧处理之前的状态。处于LOST状态，上一帧追踪失败，当前帧进行重定位
            return;
    }
    else //系统完成初始化,跟踪帧
    {
        // System is initialized. Track Frame.
        bool bOK;//bOK为临时变量，用于表示每个函数是否执行成功

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)运用运动模型或重定位初始化相机位姿估计
        // 在viewer中有个开关menuLocalizationMode，有它控制是否ActivateLocalizationMode，并最终管控mbOnlyTracking
        // mbOnlyTracking等于false表示正常VO模式（有地图更新），mbOnlyTracking等于true表示用户手动选择定位模式
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            // 正常初始化成功
            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查并更新上一帧被替换的MapPoints
                // 更新Fuse函数和SearchAndFuse函数替换的MapPoints
                CheckReplacedInLastFrame();

                // 跟踪上一帧或者参考帧或者重定位
                // 运动模型是空的或刚完成重定位
                // mCurrentFrame.mnId<mnLastRelocFrameId+2表示刚重定位少于两帧
                // 应该只要mVelocity不为空，就优先选择TrackWithMotionModel
                // mnLastRelocFrameId上一次重定位的那一帧
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 将上一帧的位姿作为当前帧的初始位姿
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();//跟踪参考帧,调用TrackReferenceKeyFrame()函数
                }
                else
                {
                    bOK = TrackWithMotionModel();//根据固定运动速度模型预测当前帧的位姿
                    if(!bOK) // TrackReferenceKeyFrame是跟踪参考帧，不能根据固定运动速度模型预测当前帧的位姿态，通过bow加速匹配（SearchByBow）
                        // 最后通过优化得到优化后的位姿
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();//重定位成功与否
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated(局部建图不工作)

            // 只进行跟踪tracking，局部地图不工作
            // 跟踪上一帧或者参考帧或者重定位
            // tracking跟丢了
            if(mState==LOST)
            {
                bOK = Relocalization();//判断重定位成功与否标志
            }
            else
            // mbVO是mbOnlyTracking为true时的才有的一个变量
            // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
            // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    // mbVO为flase则表明此帧匹配了很多的3D map点，非常好
                    if(!mVelocity.empty())//运动模型不为空的话,则用运动模型追踪
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    // mbVO为true，则表明此帧匹配了很少的3D map点，少于10个，要跪的节奏，既做跟踪又做定位

                    //我们跟踪和重定位都计算，如果重定位成功就选择重定位来计算位姿
                    bool bOKMM = false;//运动模型是否成功判断标志
                    bool bOKReloc = false;//重定位是否成功判断标志
                    vector<MapPoint*> vpMPsMM;//记录地图点
                    vector<bool> vbOutMM;//记录外点
                    cv::Mat TcwMM;  //变换矩阵
                    if(!mVelocity.empty())//判断运动模型是否存在
                    {
                        bOKMM = TrackWithMotionModel();//用运动模型追踪
                        vpMPsMM = mCurrentFrame.mvpMapPoints;//记录地图点
                        vbOutMM = mCurrentFrame.mvbOutlier;//记录外点
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();//用重定位

                    // 重定位没有成功，但是运动模型跟踪成功
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)  // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                        {
                            // 更新当前帧的MapPoints被观测程度
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)// 只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）
                    {
                        mbVO = false;// mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常，
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF; //将最新的关键帧作为referenceframe.

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // 重定位成功
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                //更新恒速运动模型TrackWithMotionModel中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;  // 其实就是Tcl
            }
            else
                mVelocity = cv::Mat();

            //清除UpdateLastFrame中为当前帧临时添加的MapPoints
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // 排除UpdateLastFrame函数中为了跟踪增加的MapPoints
                    if(pMP->Observations()<1)
                    {

                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // 步骤2.5：清除临时的MapPoints，这些MapPoints在TrackWithMotionModel的UpdateLastFrame函数里生成（仅双目和rgbd）
            // 步骤2.4中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中

            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();//将当前帧构造成关键帧,将当前关键帧设置为当前帧的参考关键帧,将关键帧插入mlNewKeyFrames中,把当前帧的信息更新到最新关键帧中。

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // 删除那些在bundle adjustment中检测为outlier的3D map点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);//当前帧保存为上一帧
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // 步骤3：记录位姿信息，用于轨迹复现
    if(!mCurrentFrame.mTcw.empty())
    {
        //计算相对姿态T_currentFrame_referenceKeyFrame
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()  //重置参考帧函数实现
{

    //如果单目初始器还没有被创建，则创建单目初始器
    if(!mpInitializer)
    {
        // Set Reference Frame
        //step 1：第一次进入该方法,如果当前帧关键点数>100,将当前帧保存为初始帧和最后一帧，并创建一个初始化器
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            //mvbPrevMatched的大小设置为已经提取的关键点的个数，mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                //mvbPrevMatched中存储关键点的坐标，用于新的一帧到来时进行匹配
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;
            //由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            //mvIniMatches中所有的元素值设置为-1
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else//如果是第二次进入，这时候已经创建了初始器
    {
        // Try to initialize
        /**
         * step 2：第二次进入该方法的时候，已经有初始化器了，如果当前帧中的关键点数>100，则继续进行匹配工作。
         * 如果当前帧特征点太少，释放初始化器。因此只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
         */
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            //特征点数少于100，此时删除初始化器
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);

        /**
         * step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
         * mvbPrevMatched为前一帧的特征点的坐标，存储了mInitialFrame中哪些点将进行接下来的匹配
         * mvIniMatches用于存储mInitialFrame,mCurrentFrame之间匹配的特征点
        */
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)三角化对应矩阵
        //上面代码模块就是，找到连续的两帧，他们满足特征匹配的约束。如果，不是连续的两帧 那么就重新开始


        //step 4：利用匹配的关键点信息进行单应矩阵和基础矩阵的计算，进而计算出相机位姿的旋转矩阵和平移矩阵
        //在该函数中创建了计算单应矩阵和基础矩阵的两个线程，计算的旋转和平移量值存放在Rcw和tcw中
        //mvIniMatches[i]中i为前一帧匹配的关键点的index，值为当前帧的匹配的关键点的index
        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))  //如果找到这样的两帧，那么就开始 初始化进程。在initializer中，得到Rcw和Tcw,如果Initialize()函数返回值为True,则 .
        {
            //step 5：删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                //判断该点是否可以三角化
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    //表示两帧对应的关键点不再匹配
                    mvIniMatches[i]=-1;
                    //关键点匹配个数-1
                    nmatches--;
                }
            }

            // Set Frame Poses
            //将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵(设定一个矩阵,默认不给赋值,里面的元素是自动为1?)
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            /**
             * step 6：由旋转矩阵和平移矩阵构造变换矩阵
             * 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
             * 这里构造出来的Tcw为一个4*4的矩阵，其中的Rcw为3*3，tcw为3*1如下所示：
             * |Rcw  tcw|
             * |0     1 |
            */

            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            /**
             * step 7：将三角化得到的3D点包装成MapPoints，在地图中显示
             * Initialize函数会得到mvIniP3D，
             * mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
             * CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            */
            CreateInitialMapMonocular();
        }
    }
}
/**
 * 功能：为关键帧初始化生成对应的MapPoints
*/
//将三角化得到的3D点包装成MapPoints( Initialize函数会得到mvIniP3D，mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
// CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中)。
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);//创建初始帧,声明指针的时候一定要给指针,new自动分配一个KeyFrame类型的地址
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);//创建关键帧


    //1：计算初始化关键帧和当前关键帧的BOW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    //2.将基础关键帧和当前关键帧插入地图中,地图中就会显示
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    //3.遍历所有匹配的关键点创建对应的mapPoint、
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        //用已经初始化好的3D点来创建world坐标
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);  // 3.1：用3D点构造MapPoint

        //i为对应关键点的index值
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        /**
         * 3.2：为该MapPoint添加相关属性
         */
        //1）.哪些关键帧可以观测到该MapPoint
        pMP->AddObservation(pKFini,i);
        //mvIniMatches[i]为pMP这个MapPoint在pKFcur这个关键帧中对应的关键点的index值
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        /**
         * 由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要判断是否更新当前点的最适合的描述子,先获得当前点的所有描述子，
         * 然后计算描述子之间的两两距离，最好的描述子与其他描述子应该具有最小的距离中值,遍历观测到3d点的所有关键帧，获得orb描述子，
         * 并插入到vDescriptors中,获得这些描述子两两之间的距离,依次找到各个描述子到其它所有所有描述子之间的距离，每一组都获得中值，
         * 寻找最小的中值，最好的描述子，该描述子相对于其他描述子有最小的距离中值，简化来讲，中值代表了这个描述子到其它描述子的平均距离，
         * 最好的描述子就是和其它描述子的平均距离最小。最后返回此描述子
         */
        //2).从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        /**
         * 更新平均观测方向以及观测距离范围，由于一个MapPoint会被许多相机观测到，因此在插入关键帧后，需要更新相应变量，获得观测到该3d点的所有关键帧，
         * 对所有关键帧对该点的观测方向归一化为单位向量进行求和，除以所有关键帧数就是获得的平均观测方向。获得观测到该点的参考关键帧和3d点在世界坐标系中的位置，
         * 得到该点到参考关键帧相机的距离，预测其在金字塔中的层数，就可以获得其其距离范围。
         */
        //3).更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        //将mappoint插入到地图中
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    //4：更新关键帧间的连接关系
    //在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    //mpMap->MapPointsInMap()返回的是map中的mappoint数目
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    //5：进行BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1      //6：将MapPoints的中值深度归一化到1，并归一化两帧之间变换,评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    //将Tc2w中的平移向量t进行了修改,利用invMedianDepth将z归一化到1
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    // 把3D点的尺度也归一化到1
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    //往LocalMapper中插入关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;//将当前关键帧设置为参考关键帧
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);//更新mLastFrame

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());//设置当前帧的相机位姿

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;    //初始化完成，修改标记值
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();//计算当前帧的Bow向量

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    //通过特征点的BoW加速匹配当前帧与参考关键帧之间的特征点
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)//匹配数小于15，表示跟踪失败
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);//将上一帧的位姿作为当前帧位姿的初始值，在PoseOptimization可以收敛快一些

    Optimizer::PoseOptimization(&mCurrentFrame);//通过优化3D-2D的重投影误差来获得位姿

    // Discard outliers
    int nmatchesMap = 0;
    //剔除优化后的outlier匹配点
    //遍历mCurrentFrame每个特征点
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])//如果这个特征点与相对应的mappoint
        {
            if(mCurrentFrame.mvbOutlier[i]) //如果这个mappoint在上次优化中被标记为outlier，则剔除
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}
/**
  * @brief 根据匀速度模型对上一帧mLastFrame的MapPoints与当前帧mCurrentFrame进行特征点跟踪匹配
  *
  * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）
  * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
  * 3. 根据匹配优化当前帧的姿态
  * 4. 根据姿态剔除误匹配
  * @return 如果匹配数大于10，返回true
  */

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);//设定当前帧与初始帧的匹配关系

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 对于双目或rgbd摄像头，根据深度值为上一关键帧生成新的MapPoints
    // （跟踪过程中需要将当前帧与上一帧进行特征点匹配，将上一帧的MapPoints投影到当前帧可以缩小匹配范围）
    // 在跟踪过程中，去除outlier的MapPoint，如果不及时增加MapPoint会逐渐减少
    // 这个函数的功能就是补充增加RGBD和双目相机上一帧的MapPoints数
    UpdateLastFrame();
//根据Const Velocity Model(认为这两帧之间的相对运动和之前两帧间相对运动相同)估计当前帧的粗略位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    //对上一帧的MapPoints进行跟踪，看上一帧能看到的mappoint对应的当前帧哪些特征点
    // 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    //匹配数量太少，扩大特征匹配搜索框重新进行mappoint跟踪
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);// 优化位姿

    // Discard outliers
    int nmatchesMap = 0;
    //上一步的位姿优化更新了mCurrentFrame的outlier，需要将mCurrentFrame的mvpMapPoints更新
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }//如果当前帧可以看到的mappoint同时能被其他keyframe看到
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();// 【1】首先对局部地图进行更新(UpdateLocalMap) 生成对应当前帧的 局部地图

    SearchLocalPoints();// 局部地图点 搜寻和当前帧 关键点描述子 的匹配 有匹配的加入到 当前帧 特征点对应的地图点中

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)//更新地图点的状态
    {
        if(mCurrentFrame.mvpMapPoints[i]) //是当前帧的地图点
        {
            if(!mCurrentFrame.mvbOutlier[i])//是内点,符合变换关系
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)//刚刚进行过重定位,则需要匹配点对数大于50才认为成功
        return false;

    if(mnMatchesInliers<30)//正常情况下找到的匹配点对数大于30算成功
        return false;
    else
        return true;
}

    // 步骤1：如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)  //只追踪不建图的过程也是不需要关键帧的
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())//建图线程停止
        return false;

    const int nKFs = mpMap->KeyFramesInMap();  //地图中关键帧的数量

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // 步骤2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    // 如果关键帧比较少，则考虑插入关键帧
    // 或距离上一次重定位超过1s，则考虑插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    // 步骤3：得到参考关键帧跟踪到的MapPoints数量
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // 步骤4：查询局部地图管理器是否繁忙
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // 步骤5：对于双目或RGBD摄像头，统计总的可以添加的MapPoints数量和跟踪到地图中的MapPoints数量
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);
    // 步骤6：决策是否需要插入关键帧
    // Thresholds
    // Thresholds 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f; //关键帧只有一帧,那么插入关键帧的阈值设置很低

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames; //很长时间没有插入关键帧
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle); // localMapper处于空闲状态
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose);	// 跟踪要跪的节奏,0.25和0.3是一个比较低的阈值
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);// 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}
//调用CreateNewKeyFrame()函数来创建关键帧
void Tracking:: CreateNewKeyFrame() //对于非单目的情况,需要同时创建关键帧
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // 步骤1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // 步骤2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 这段代码和UpdateLastFrame中的那一部分代码功能相同
    // 步骤3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // 步骤3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // 步骤3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}
/**
 * @brief 对Local MapPoints进行跟踪
 * 搜索 在对应当前帧的局部地图内搜寻和 当前帧地图点匹配点的 局部地图点
 * 局部地图点 搜寻和当前帧 关键点描述子 的匹配 有匹配的加入到 当前帧 特征点对应的地图点中
 *
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 步骤1：遍历当前帧的mvpMapPoints，标记这些MapPoints,它们不参与之后的搜索,因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();// 更新能观测到该点的帧数加1
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;// 标记该点被当前帧观测到
                pMP->mbTrackInView = false;// 标记该点将来不被投影，因为已经匹配过
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // 步骤2：将所有局部MapPoints投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) // 已经被当前帧观测到MapPoint不再判断是否能被当前帧观测到
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))// 步骤2.1：判断LocalMapPoints中的点是否在在视野内
        {
            pMP->IncreaseVisible();// 观测到该点的帧数加1，该MapPoint在某些帧的视野范围内
            nToMatch++; // 只有在视野范围内的MapPoints才参与之后的投影匹配
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2) // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th); // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
    }
}

void Tracking::UpdateLocalMap()//更新局部地图,用于局部地图的跟踪,关键帧+地图点
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();//更新关键帧,共视化程度高的关键帧
    UpdateLocalPoints();//更新地图点,局部关镇帧对应的地图点就是地图点
}
    //局部地图点的更新比较容易,完全根据局部关键帧来,所有局部关键帧的地图点就构成局部地图点
    //局部关键帧mvpLocalKeyFrames的MapPoints,更新mvpLocalMapPoints
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;//每一个局部关键帧
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();// 局部关键帧的地图点
        // 每一个局部关键帧的地图点
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;//每一个局部地图点
            if(!pMP)//空的点直接跳过
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)//已经更新过了
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP); //更新
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

//遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新mvpLocalKeyFrames
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 更新地图点的观测帧
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i]) //当前帧的地图点
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;// 地图点的观测帧 观测地图点次数++
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;//未观测到,地图点清除
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    //  1. 共视化程度高的关键帧观测到当前帧地图点次数多的关键帧；
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;//地图点的关键帧

        if(pKF->isBad())
            continue;

        if(it->second>max) //观测到地图点数量最多的关键帧
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);// 保存局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 始终限制关键数量不超过80
            break;

        KeyFrame* pKF = *itKF;
        // 根据权重w 二分查找有序序列中的某些对象
        // 返回前w个有序关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);// 加入局部关键帧
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
// 2. 子关键帧；
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }
// 3. 父关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);//找到与当前帧很相似的候选关键帧

    if(vpCandidateKFs.empty())//如果候选关键帧为空，则返回Relocalization失败
        return false;

    const int nKFs = vpCandidateKFs.size();//候选帧的数目

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    //表示各个候选帧的mappoint与和当前帧特征点的匹配
    //现在你想把mCurrentFrame的特征点和mappoint进行匹配，有个便捷的方法就是，
    //让mCurrentFrame特征点和候选关键帧的特征点进行匹配,然后我们是知道候选关键帧特征点与mappoint的匹配的
    //这样就能够将mCurrentFrame特征点和mappoint匹配起来了，相当于通过和候选关键帧这个桥梁匹配上了mappoint
    //vvpMapPointMatches[i][j]就表示mCurrentFrame的第j个特征点如果是经由第i个候选关键帧匹配mappoint，是哪个mappoint
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;//bool类型的容器.里面存放的值为true和false
    vbDiscarded.resize(nKFs);

    int nCandidates=0;


    //候选帧和当前帧进行特征匹配，剔除匹配数量少的候选关键帧
    //为未被剔除的关键帧就新建PnPsolver，准备在后面进行epnp
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];//候选帧
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            //mCurrentFrame与候选关键帧进行特征点匹配
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)//剔除
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {

                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    //大概步骤是这样的，小循环for不断的遍历剩下的nCandidates个的候选帧，这些候选帧对应有各自的PnPsolvers
    //第i次for循环所对应的vpPnPsolvers[i]就会执行5次RANSAC循环求解出5个位姿。
    //通过计算5个位姿对应的匹配点的inliner数量来判断位姿的好坏。如果这5个位姿比记录中的最好位姿更好，更新最好位姿以及对应的匹配点哪些点是inliner
    //如果最好的那个位姿inliner超过阈值，或者vpPnPsolvers[i]RANSAC累计迭代次数超过阈值，都会把位姿拷贝给Tcw。否则Tcw为空
    //如果Tcw为空，那么就循环计算下一个vpPnPsolvers[i+1]
    //通过5次RANSAC求解位姿后，如果Tcw不为空，这继续判断它是否和当前帧匹配.
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)//遍历候选帧
        {
            if(vbDiscarded[i]) //vbDiscarded[i]里面是存放不合格的候选帧
                continue;

            // Perform 5 Ransac Iterations
            //此次RANSAC会计算出一个位姿，在这个位姿下，mCurrentFrame中的特征点哪些是有mappoint匹配的，也就是哪些是inliner
            //vbInliers大小是mCurrentFrame中的特征点数量大小
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);  //通过EPnP算法估计姿态，有5次RANSAC循环

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            //相机姿态算出来有两种情况，一种是在RANSAC累计迭代次数没有达到mRansacMaxIts之前，找到了一个复合要求的位姿
            //另一种情况是RANSAC累计迭代次数到达了最大mRansacMaxIts
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();//np为mCurrentFrame的特征点数量

                //根据vbInliers更新mCurrentFrame.mvpMapPoints，也就是根据vbInliers更新mCurrentFrame的特征点与哪些mappoint匹配
                //并记下当前mCurrentFrame与哪些mappoint匹配到sFound，以便后面快速查询
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame); //BA优化位姿

                if(nGood<10)
                    continue;
//剔除PoseOptimization算出的mvbOutlier
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)// 如果内点较少,设想一个大点窗口重新优化
                {
                    // mCurrentFrame中特征点已经匹配好一些mappoint在sFound中，如果内点较少,mCurrentFrame想要更多的mappoint匹配
                    //于是通过matcher2.SearchByProjection函数将vpCandidateKFs[i]的mappoint悉数投影到CurrentFrame再就近搜索特征点进行匹配
                    //mCurrentFrame返回得到通过此函数匹配到的新mappoint的个数
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)  //如果目前mCurrentFrame匹配到的mappoint个数超过50
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);//优化位姿，返回(nadditional+nGood)有多少点是inliner

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)//如果nGood不够多，那缩小搜索框重复再匹配一次
                        {
                            //更新sFound，也就是目前mCurrentFrame与哪些mappoint匹配
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            //缩小搜索框重复再匹配一次,返回这个新得到的匹配数
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
