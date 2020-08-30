let video;
let poseNetSource;
let poseSource;
let poseCapture;
let skeletonSource;
let skeletonCapture;
let stream;
let width = window.innerWidth;
let height = window.innerHeight;

// This function is called when the video loads
function vidLoad() {
  video.loop();
  video.volume(0);
}

function setup(){
    
    createCanvas(width, height);

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
    // mirror video
    translate(video.width, 0);
    scale(-1, 1);
    // draw the webcam stream

    let windowOffset = -width/2 + 340;

    image(stream, windowOffset, 0, 680, 480);

    // draw poses
    if(poseSource){
        
        let offsetX;
        let offsetY;

        //lets the model follow the stream, allows the source video skeleton to follow the webcam user
        if(poseCapture){
            offsetX = poseSource.nose.x - poseCapture.nose.x - windowOffset;
            offsetY = 0.5 * (poseSource.leftShoulder.y - poseCapture.leftShoulder.y + poseSource.rightShoulder.y - poseCapture.rightShoulder.y);
        }

        // draw joints at circles
        for (let i = 0; i < poseSource.keypoints.length; i++){
            let x = poseSource.keypoints[i].position.x - offsetX;
            let y = poseSource.keypoints[i].position.y - offsetY;
            if(poseSource.keypoints[i].score > 0.5)
            fill(255);
            ellipse(x,y,5,5);
        }

        // draw connections between joints
        for (let i = 0; i < skeletonSource.length; i++) {
            let a = skeletonSource[i][0];
            let b = skeletonSource[i][1];
            strokeWeight(5);
            stroke(255);
            line(a.position.x - offsetX, a.position.y - offsetY, b.position.x - offsetX, b.position.y - offsetY);      
        }
    }
}

