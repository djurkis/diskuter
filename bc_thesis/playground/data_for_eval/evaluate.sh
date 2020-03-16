#!/usr/bin/bash

for file in predictions/*
do
	files2rouge ${file} test.tgt > ${file}.out &
done
