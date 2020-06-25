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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include <unistd.h>
#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // 步骤1：设置进程间的访问标志告诉Tracking线程,LocalMapping线程正在处理新的关键帧,处于繁忙状态
        // LocalMapping线程处理的关键帧都是Tracking线程发过的
        // 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快(此时不是不接受新的关键帧吗)
        SetAcceptKeyFrames(false);//为false的时候表示暂时LocalMapping线程正在工作,暂时不接受新的关键帧

        // Check if there are keyframes in the queue
        // 等待处理的关键帧列表不为空
        if(CheckNewKeyFrames())  //CheckNewKeyFrames()为true时,表示待处理的关键帧列表不为空
        {
            // BoW conversion and insertion in Map
            // 步骤2：计算关键帧特征点的词典单词向量BoW映射,处理关键帧,将关键帧插入地图
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();//调用该函数去删除不符合要求的MapPoints.

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())//再次检查关键帧序列,知道最后一帧处理完成之后.
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();//调用该函数进行冗余MapPoint融合
            }

            mbAbortBA = false;
// 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
// 步骤6：局部地图优化
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
// 步骤7： 关键帧融合检测并剔除当前帧相邻的关键帧中冗余的关键帧
// 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到
// 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密,在这里再删除冗余的关键帧
                KeyFrameCulling();//调用该函数删除冗余关键帧
            }
// 步骤8：将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }// 步骤9：等待线程空闲 完成一帧关键帧的插入融合工作
        else if(Stop())//如果建图过程被暂停且没有完成,那么线程休眠3ms
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())//建图完成退出循环
                break;
        }

        ResetIfRequested();// 检查重置

        // Tracking will see that Local Mapping is busy
        // 步骤10：告诉Tracking 线程Local Mapping线程空闲可一处理接收 下一个 关键帧
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        // 步骤1：从缓冲队列中取出一帧待处理的关键帧
        // Tracking线程向LocalMapping中插入关键帧存在该队列中
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();//从列表中获得一个等待被插入的关键帧
        mlNewKeyFrames.pop_front();// pop_front()删除第一个元素
    }

    // Compute Bags of Words structures
    // 步骤2：计算该关键帧特征点的Bow映射关系
    //  根据词典计算当前关键帧Bow，便于后面三角化恢复新地图点
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // 当前关键帧TrackLocalMap中跟踪局部地图匹配上的地图点
    // 步骤3：跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
    // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配,
    // 但没有对这些匹配上的MapPoints与当前帧进行关联
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i]; //每一个与当前关键帧匹配好的地图点
        if(pMP)//地图点存在
        {
            if(!pMP->isBad())
            {
                // 为当前帧在tracking过程跟踪到的MapPoints更新属性
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))//如果pMP是不在关键帧mpCurrentKeyFrame中，则更新mappoint的属性
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);// 地图点添加关键帧
                    pMP->UpdateNormalAndDepth();// 地图点更新平均观测方向和观测距离深度
                    pMP->ComputeDistinctiveDescriptors();// 加入关键帧后,更新地图点的最佳描述子
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);// 候选待检查地图点存放在mlpRecentAddedMapPoints
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();// 步骤4：更新关键帧间的连接关系，Covisibility图和Essential图(tree)

    // Insert Keyframe in Map
    //更新好了MapPoint和关键帧的相关关系后，接下来就是调用下面这句代码将关键帧插入地图点中。
    mpMap->AddKeyFrame(mpCurrentKeyFrame);// 步骤5：将该关键帧插入到地图中
}
//删除不符合要求的地图点
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();//待检测的地图点迭代器
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;//当前关键帧的id

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end()) // 遍历等待检查的地图点MapPoints
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())//地图点是否是好的
        {
            lit = mlpRecentAddedMapPoints.erase(lit);//步骤1：已经是坏点的MapPoints直接从检查链表中删除
        }
        //跟踪(匹配上)到该地图点的普通帧帧数（IncreaseFound)<应该观测到该地图点的普通帧数量（25%*IncreaseVisible）
        //该地图点虽在视野范围内，但很少被普通帧检测到,剔除.
        else if(pMP->GetFoundRatio()<0.25f )
        {
            // 步骤2：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件1：
            // 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
            // IncreaseFound() / IncreaseVisible(该地图点在视野范围内) < 25%，注意不一定是关键帧。
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);//从list列表删除
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            // 步骤3：将不满足VI-B条件的MapPoint剔除
            // VI-B 条件2：从该点建立开始，到现在已经过了不小于2帧，
            // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            // 步骤4：从建立该点开始，已经过了3帧(前三帧地图点比较宝贵需要特别检查)，放弃对该MapPoint的检测
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}
/**
 * @brief 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
 *  根据当前关键帧恢复出一些新的地图点，不包括和当前关键帧匹配的局部地图点（已经在ProcessNewKeyFrame中处理）
 *  先处理新关键帧与局部地图点之间的关系，然后对局部地图点进行检查，
 *  最后再通过新关键帧恢复 新的局部地图点：CreateNewMapPoints()
 *
 * 步骤1：在当前关键帧的 共视关键帧 中找到 共视程度 最高的前nn帧 相邻帧vpNeighKFs
 * 步骤2：遍历和当前关键帧 相邻的 每一个关键帧vpNeighKFs
 * 步骤3：判断相机运动的基线在（两针间的相机相对坐标）是不是足够长
 * 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵 F =  inv(K1 转置) * t12 叉乘 R12 * inv(K2)
 * 步骤5：通过帧间词典向量加速匹配，极线约束限制匹配时的搜索范围，进行特征点匹配
 * 步骤6：对每对匹配点 2d-2d 通过三角化生成3D点,和 Triangulate函数差不多
 *  步骤6.1：取出匹配特征点
 *  步骤6.2：利用匹配点反投影得到视差角   用来决定使用三角化恢复(视差角较大) 还是 直接2-d点反投影(视差角较小)
 *  步骤6.3：对于双目，利用双目基线 深度 得到视差角
 *  步骤6.4：视差角较大时 使用 三角化恢复3D点
 *  步骤6.4：对于双目 视差角较小时 二维点 利用深度值 反投影 成 三维点    单目的话直接跳过
 *  步骤6.5：检测生成的3D点是否在相机前方
 *  步骤6.6：计算3D点在当前关键帧下的重投影误差  误差较大跳过
 *  步骤6.7：计算3D点在 邻接关键帧 下的重投影误差 误差较大跳过
 *  步骤6.9：三角化生成3D点成功，构造成地图点 MapPoint
 *  步骤6.9：为该MapPoint添加属性
 *  步骤6.10：将新产生的点放入检测队列 mlpRecentAddedMapPoints  交给 MapPointCulling() 检查生成的点是否合适
 * @see
 */
    void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;//共视程度高的帧的数量(双目/深度相机)
    if(mbMonocular)
        nn=20;
    // 步骤1：在当前关键帧的共视关键帧中找到共视程度最高的nn帧相邻帧vpNeighKFs
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);// 描述子匹配器
// 当前关键帧旋转平移矩阵向量
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();//旋转矩阵
    cv::Mat Rwc1 = Rcw1.t();//旋转矩阵转置
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();//平移向量
    cv::Mat Tcw1(3,4,CV_32F);//变换矩阵
    //以下两步操作就是利用旋转和平移构成变换矩阵
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();//得到当前关键帧的光心在世界坐标系的坐标

    //相机内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)// 步骤2：遍历和当前关键帧相邻的每一个关键帧vpNeighKFs
    {
        if(i>0 && CheckNewKeyFrames())//如果有关键帧在检测队列中,返回进行局部地图构建
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];//邻接的关键帧

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();//邻接的关键帧在世界坐标系中的坐标
        cv::Mat vBaseline = Ow2-Ow1;//基线向量,两个关键帧间的相机相对坐标
        const float baseline = cv::norm(vBaseline);//基线长度
// 步骤3：判断相机运动的基线是不是足够长
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)// 如果是立体相机,关键帧间距太小时不生成3D点
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2); //邻接关键帧的场景深度中值
            const float ratioBaselineDepth = baseline/medianDepthKF2;// baseline与景深的比例

            if(ratioBaselineDepth<0.01) // 如果特别远(比例特别小)，那么不考虑当前邻接的关键帧，不生成3D点
                continue;
        }

        // Compute Fundamental Matrix
        // 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵
        // 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
        // F =  inv(K1 转置)*E*inv(K2) = inv(K1 转置) * t12 叉乘 R12 * inv(K2)
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);//K1和K2分别是相机的内参吧

        // Search matches that fullfil epipolar constraint
        // 步骤5：通过帧间词典向量加速匹配,极线约束限制匹配时的搜索范围,进行特征点匹配
        vector<pair<size_t,size_t> > vMatchedIndices;// 特征匹配候选点
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);//三角化地图点

        cv::Mat Rcw2 = pKF2->GetRotation();//相邻关键帧旋转矩阵
        cv::Mat Rwc2 = Rcw2.t();//旋转矩阵转置
        cv::Mat tcw2 = pKF2->GetTranslation();//平移向量
        cv::Mat Tcw2(3,4,CV_32F);//定义一个3*4的变换矩阵
        //用旋转矩阵和平移向量构造变换矩阵
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        // 三角化每一个匹配点对
//步骤6：对每对匹配点 2d-2d 通过三角化生成3D点,和Triangulate函数差不多
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // 步骤6.1：取出匹配特征点
            const int &idx1 = vMatchedIndices[ikp].first;// 当前匹配对在当前关键帧中的索引
            const int &idx2 = vMatchedIndices[ikp].second;// 当前匹配对在邻接关键帧中的索引

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];//当前关键帧的特征点
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];//右图像匹配点横坐标
            bool bStereo1 = kp1_ur>=0;//右图像匹配点横坐标>=0是双目/深度相机
            //这是邻接关键帧的特征点和右图像匹配点的横坐标
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            // 步骤6.2：利用匹配点反投影得到视差角
            // 相机归一化平面上的点坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            //由相机坐标系转到世界坐标系,得到视差角余弦值
            cv::Mat ray1 = Rwc1*xn1;// 相机坐标系 ------> 世界坐标系
            cv::Mat ray2 = Rwc2*xn2;
            // 向量乘积：a * b = |a|*|b|*cos<a,b> 推出：cos<a,b> = (a * b)/(|a|*|b|)
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));//视差角余弦

            float cosParallaxStereo = cosParallaxRays+1;// 加1是为了让cosParallaxStereo随便初始化为一个很大的值
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;
// 步骤6.3：对于双目,利用双目基线深度得到视差角
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);// 得到双目观测的视差角
// 步骤6.4：三角化恢复3D点
            cv::Mat x3D;
            // 视差角度小时用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))//表明视差角正常(为什么?)
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);//其次点坐标,去除尺度

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)//步骤6.4：对于双目视差角较小时二维点利用深度值反投影成三维点
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1); //二维点反投影成三维点
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else// 单目视差角较小时生成不了三维点
                continue; //No stereo and very low parallax 没有双目/深度且两针视角差太小 三角测量也不合适得不到三维点

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);// 步骤6.5：检测生成的3D点是否在相机前方
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // 步骤6.6：计算3D点在当前关键帧下的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];//误差分布参数
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);//相机归一化坐标(应该是把3D空间点通过旋转和平移到相机坐标系下)
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {//单目
                float u1 = fx1*x1*invz1+cx1;//计算像素坐标
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;//投影误差过大
            }
            else
            {// 双目/深度相机,有右图像匹配点横坐标差值
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;//左图像坐标值 - 视差 = 右图像匹配点横坐标
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                // 基于卡方检验计算出的阈值（假设测量有一个一个像素的偏差）
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
           // 步骤6.7：计算3D点在 邻接关键帧 下的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // 步骤6.8：检查尺度连续性
            cv::Mat normal1 = x3D-Ow1;//世界坐标系下,3D点与相机间的向量,方向由相机指向3D点
            float dist1 = cv::norm(normal1);//向量取模长

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;//ratioDist是不考虑金字塔尺度下的距离比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];//金字塔尺度因子的比例

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)//深度比值和两幅图像下的金字塔层级比值应该相差不大
                continue;

            // Triangulation is succesfull
            // 步骤6.9：三角化生成3D点成功，构造成地图点 MapPoint
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
// 步骤6.9：为该MapPoint添加属性：观测到该MapPoint的关键帧,该MapPoint的描述子,该MapPoint的平均观测方向和深度范围,添加地图点到地图
            pMP->AddObservation(mpCurrentKeyFrame,idx1);//地图点添加观测帧
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);// 关键帧添加地图点
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();//计算地图点的描述子

            pMP->UpdateNormalAndDepth();//更新深度

            mpMap->AddMapPoint(pMP);//增加地图点
            // 步骤6.10：将新产生的点放入检测队列 mlpRecentAddedMapPoints
            // 这些MapPoints都会经过MapPointCulling函数的检验
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // 步骤1：获得当前关键帧在covisibility图中权重排名前nn的一级邻接关键帧,找到当前帧一级相邻与二级相邻关键帧
    int nn = 10;
    if(mbMonocular)
        nn=20;//单目 多找一些
    // 一级相邻
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;//最后合格的一级二级相邻关键帧
    // 遍历每一个一级相邻帧
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;//一级相邻关键帧
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)//坏帧或已经加入过
            continue;
        vpTargetKFs.push_back(pKFi);// 加入 最后合格的相邻关键帧
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;//已经做过相邻匹配,标记已经加入的关键帧

        // Extend to some second neighbors
        // 步骤2：获得当前关键帧在 其一级相邻帧的  covisibility图中权重排名前5的二级邻接关键帧(同上)
        //二级相邻(与一级相邻帧的共视帧)
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

// 步骤3：将当前帧的地图点MapPoints分别与其一级二级相邻帧的地图点MapPoints进行融合
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)//遍历合格的一级二级关键帧
    {
        KeyFrame* pKFi = *vit;
        // 投影当前帧的MapPoints到相邻关键帧pKFi中，在附加区域搜索匹配关键点，并判断是否有重复的MapPoints
        // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
        // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
        matcher.Fuse(pKFi,vpMapPointMatches); //地图点融合函数
    }

// 步骤4：将一级二级相邻帧所有的地图点MapPoints 与当前帧（的MapPoints）进行融合
    // 遍历每一个一级邻接和二级邻接关键帧 找到所有的地图点
    // Search matches by projection from target KFs in current KF
    // 用于存储一级邻接和二级邻接关键帧所有MapPoints的集合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());// 帧数量 × 每一帧地图点数量

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;//标记已经加入的
            vpFuseCandidates.push_back(pMP);//加入一级二级相邻帧地图点集合
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    // 步骤5：更新当前帧MapPoints的描述子，深度，观测主方向等属性
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();//当前帧所有的匹配地图点
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];//当前帧每个关键点匹配的地图点
        if(pMP)//匹配的地图点存在
        {
            if(!pMP->isBad())//非坏点
            {
                pMP->ComputeDistinctiveDescriptors(); //更新地图点的描述子(在所有观测在的描述子中选出最好的描述子)
                pMP->UpdateNormalAndDepth(); //更新平均观测方向和观测距离
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();//更新当前帧与其他帧的连接关系
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;//内参
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}
/**
 * @brief    关键帧剔除
 *  在Covisibility Graph 关键帧连接图 中的关键帧，
 *  其90%以上的地图点MapPoints能被其他关键帧（至少3个）观测到，
 *  则认为该关键帧为冗余关键帧。
 * @param  pKF1 关键帧1
 * @param  pKF2 关键帧2
 * @return 两个关键帧之间的基本矩阵 F
 */
    void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points

    // 步骤1：根据Covisibility Graph 关键帧连接 图提取当前帧的 所有共视关键帧(关联帧)
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
//对上一步提取的关键帧进行遍历
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {

        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)//第一帧关键帧为初始化世界关键帧 跳过
            continue;
        //步骤2：提取每个共视关键帧的地图点 MapPoints
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();// 局部关联帧匹配的地图点

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
// 步骤3：遍历该局部关键帧的MapPoints，判断是否90%以上的MapPoints能被其它关键帧（至少3个）观测到
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];// 该局部关键帧的地图点 MapPoints
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)//双目或者深度相机
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)//仅考虑近处的地图点
                            continue;
                    }
                    //单目的情况
                    nMPs++;
                    // 地图点 MapPoints 至少被三个关键帧观测到
                    if(pMP->Observations()>thObs)//观测个数>3  (thObs前面定义值为3)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave; //关键帧金字塔层数
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();//局部观测关键帧地图
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;//局部观测关键帧
                            if(pKFi==pKF)// 跳过 原地图点的帧
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;//观测关键帧的金字塔层数
                        //尺度约束,要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {//该mapPoint至少被3个关键帧观测到
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }
        // 步骤4：该局部关键帧90%以上的MapPoints能被其它关键帧（至少3个）观测到，则认为是冗余关键帧
        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
