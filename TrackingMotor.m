v = VideoReader(path);
NbFrame = v.Duration*v.FrameRate;

% for f = [1:NbFrame]
%     frame = read(v,f);
%     frame = imbinarize(frame, 'adaptive');
%     frame = frame(:,:,1);
%     imshow(frame);
%     pause(0.5);
% end


frame = read(v,f);
imshow(frame);
h = imrect;
bbox = wait(h);
frameROI = insertShape(frame, 'Rectangle', bbox);
imshow(frameROI);

bboxPoints = bbox2points(bbox(1, :));
points = detectMinEigenFeatures(rgb2gray(frame), 'ROI', bbox);
hold on;
plot(points);

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, frame);

%%
% Set the path
path = 'D:\TheMagneticMonster\Pierre\Data\2019-05-28\Run 04\11h14m17s.avi';
videoFileReader = vision.VideoFileReader(path);
videoFrame = step(videoFileReader);

% Set the ROI
imshow(videoFrame);
h = imrect;
bbox = wait(h);
frameROI = insertShape(videoFrame, 'Rectangle', bbox);
imshow(frameROI);
bboxPoints = bbox2points(bbox(1, :));
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
hold on;
plot(points.Location(:,1), points.Location(:,2), '*');

% Initialize the tracker with the initial point locations and the initial video frame.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
points = points.Location;
initialize(pointTracker, points, videoFrame);
[points,point_validity,scores] = pointTracker(videoFrame)

% setPoints(pointTracker, points);
% [points,point_validity,scores] = pointTracker(videoFrame)

%%
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);
%%
oldPoints = points;

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);
        
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end
    
    % Display the annotated video frame using the video player object
    %step(videoPlayer, videoFrame);
end

% Clean up
release(videoFileReader);
release(videoPlayer);