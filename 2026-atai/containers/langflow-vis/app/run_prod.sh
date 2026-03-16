#!/bin/bash

uvicorn app.server.main:app --host 0.0.0.0 --port 8080
