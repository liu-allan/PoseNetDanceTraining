let video;
let poseNetSource;
let poseSource;
let poseCapture;
let skeletonSource;
let skeletonCapture;
let stream;
let canvas;

// This function is called when the video loads
function vidLoad() {
    video.loop();
    video.volume(0);
}

function setup(){
    
    canvas = createCanvas(windowWidth, windowHeight);
    canvas.style('display', 'block');

    // capture video from webcam
    stream = createCapture(VIDEO)
    stream.hide();

    // capture video from source video
    video = createVideo(['assets/DoubleDreamFeet.mp4'],vidLoad);
    video.hide();

    // instantiate network for the source video
    poseNetSource = ml5.poseNet(video, modelLoaded);
    poseNetSource.on('pose', gotSourcePoses); 

    // instantiate network for the webcame video
    poseNetCapture = ml5.poseNet(stream, modelLoaded);
    poseNetCapture.on('pose', gotCapturePoses); 

}

function centerCanvas() {
    var x = (windowWidth - width) / 2;
    var y = (windowHeight - height) / 2;
    canvas.position(x, y);
}

// captures first pose detected from source video 
function gotSourcePoses(poses){
    if (poses.length > 0){
        poseSource = poses[0].pose;
        skeletonSource = poses[0].skeleton;
    }
}

// captures first pose detected from webcam video
function gotCapturePoses(poses){
    if (poses.length > 0){
        poseCapture = poses[0].pose;
        skeletonCapture = poses[0].skeleton;
    }
}
  
function modelLoaded(){
    console.log('poseNet ready');
}

function draw(){

    // erase previously drawn skeleton/joints
    erase();
    rect(0, 0, width, height);
    noErase();

    // mirror video
    translate(video.width, 0);
    scale(-1, 1);
    // draw the webcam stream

    // center the image
    let centerOffsetX = (windowWidth/2 - 320);
    let centerOffsetY = -(windowHeight/2 - 240);

    image(stream, -centerOffsetX, -centerOffsetY, 640, 480);

    // draw poses
    if(poseSource && poseCapture){
        
        let offsetX;
        let offsetY;

        // lets the model follow the stream, allows the source video skeleton to follow the webcam user
        // offsetX is calculated between source and captures noses
        // offsetY is calculated between source and captures shoulders

        offsetX = poseSource.nose.x - poseCapture.nose.x;
        offsetY = 0.5 * (poseSource.leftShoulder.y - poseCapture.leftShoulder.y + poseSource.rightShoulder.y - poseCapture.rightShoulder.y);
        
        // draw the joints
        for (let i = 0; i < poseCapture.keypoints.length; i++){

            let x2 = poseCapture.keypoints[i].position.x - centerOffsetX;
            let y2 = poseCapture.keypoints[i].position.y - centerOffsetY;

            let x1 = poseSource.keypoints[i].position.x - offsetX - centerOffsetX;
            let y1 = poseSource.keypoints[i].position.y - offsetY - centerOffsetY;

            var error = distanceBetweenPoints(x1,y1,x2,y2);
            
            // if model confidence is greater than 0.5, draw
            if(poseSource.keypoints[i].score > 0.5){
                // draw green if error is less than 50
                if(error < 50){
                    fill(0, 255, 0);
                // draw red if error is greater than 50
                }else{
                    fill(255, 0, 0);
                }
                // make sure the body is within frame
                if(y1 > -centerOffsetY && y1 < -centerOffsetY + 480 && x1 > -centerOffsetX && x1 < -centerOffsetX + 640){
                    ellipse(x1,y1,8);  
                }
                 
            }
        }

        // draw skeleton
        for (let i = 0; i < skeletonCapture.length && i < skeletonSource.length; i++) {

            let a = skeletonSource[i][0];
            let b = skeletonSource[i][1];

            let c = skeletonCapture[i][0];
            let d = skeletonCapture[i][1];

            let captureAngle = findAngle(c.position.x, c.position.y, d.position.x, d.position.y);
            let sourceAngle = findAngle(a.position.x, a.position.y, b.position.x, b.position.y);

            strokeWeight(4);
            // draw green if angle is within range
            if(captureAngle < sourceAngle + 0.7 && captureAngle > sourceAngle - 0.7){
                stroke(0,255,0);
            // draw red if out of range
            }else{
                stroke(255,0,0);
            }
            
            let x1 = a.position.x - offsetX;
            let y1 = a.position.y - offsetY;
            let x2 = b.position.x - offsetX; 
            let y2 = b.position.y - offsetY;

            if(y1 > -centerOffsetY && y1 < -centerOffsetY + 480 && x1 > -centerOffsetX && x1 < -centerOffsetX + 640 && y2 > -centerOffsetY && y2 < -centerOffsetY + 480 && x2 > -centerOffsetX && x2 < -centerOffsetX + 640)
                line(x1 - centerOffsetX, y1 - centerOffsetY, x2 - centerOffsetX, y2 - centerOffsetY);      
        }
    }
}

// euclidean distance between two points
function distanceBetweenPoints(x1,y1,x2,y2){
    var distance = Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2));
    return distance;
}

// arctan to calculate angle
function findAngle(x1,y1,x2,y2){
    var angle = Math.atan((y2-y1)/(x2-x1));
    return angle;
}