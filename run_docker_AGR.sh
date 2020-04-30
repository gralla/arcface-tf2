#!/bin/bash
docker run -d --runtime=nvidia \
        --name idvirtuel_tf1_AGR \
	    -v $PWD:/workspace \
		-v /mnt/data/TAE/IACIO2020/id_virtuel/:/data \
		-w /workspace \
		-p 2355-2357:2355-2357 \
		-e DISPLAY=$DISPLAY \
		idvirtuel_tf1:latest \
		tail -f /dev/null
