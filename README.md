### W251 hw7
#### Travis Metz, Tuesday 2pm PST

I got the face detector up and running using the tensorflow model.  You can see the face detector box in the output.png file in this repo.

I was not able to get the full HW3 sequence going (mosquitto broker, forwarder, cloud broker, cloud processor, etc).  I tried for quite some time, but my HW3 code was clunky and after spending many hours working on it, I decided I needed to move on to other assignments.  (I needed to better document that HW3 code and make it more self-sufficient in terms of setup scripts etc.  Lesson for the future.)

#### Notes on how it should all work

I have five containers running.  My start to finish:
- on Jetson, a l4t base image using opencv to capture faces and publish to Jetson broker
- on Jetson, an alpine base image using mosquitto to broker Messages
- on Jetson, an alphine/python base image that subscribes to broker and publishes to a cloud broker
- on IBM Cloud, an alpine base image using mosquitto to broker Messages
- on IBM Cloud, an ubuntu base image using opencv that subscribes to topic from cloud broker, converts messages to images and stores in my S3 bucket

I left mosquitto QoS at zero.  This means that, at end of pipeline, I am not storing all the faces that the face detector publishes (as it is fire and forget).

My topic is named 'faces_topic'.

#### set up docker network on jetson
```docker network create --driver bridge hw7```

#### build images if do not exist
docker build -t fd7-image -f Dockerfile.face_detector .


#### get opencv face processor running

```docker run -e DISPLAY=$DISPLAY --privileged --name fd7 --net host -v /home/trmetz/trm/hw7:/hw7 -ti fd7-image```

From within /hw7, ```python3 video_tf.py```

#### get broker running

```docker run --name mosq-broker -p 1883:1883 -v /home/trmetz/trm/hw7:/hw7 --network hw7 -ti broker-image mosquitto```

#### get forwarder running

```docker run --name forwarder --network hw7 -v /home/trmetz/trm/hw7:/hw7 -ti forwarder-image sh```

From within /hw7, ```python3 forwarder.py```

#### set up docker network in cloud
```docker network create --driver bridge hw7-cloud```

#### from cloud, start cloud broker running

```docker run --name broker --network hw7-cloud -p 1883:1883 -ti broker-image mosquitto```

#### from cloud, start cloud processor running

```docker run --name cloud_processor -v /root/w251_trm:/hw7 --privileged --network hw7-cloud -ti cloud-processor-image bash```

Set up S3 within that cloud processor:

```s3fs s3-trm /hw7/mybucket -o passwd_file=/hw7/.cos_creds -o sigv2 -o use_path_request_style -o url=https://s3.us-east.objectstorage.softlayer.net```

From within /hw7, ```python3 processor.py```


#### ssh IBM

To get to IBM VSI that is running two containers for pictures (this is from laptop, not from jumpbox VS)

```ssh root@169.62.39.215 -i .ssh/id_rsa```

When jetson  stops - restart the opencv python program - that times out with broker
