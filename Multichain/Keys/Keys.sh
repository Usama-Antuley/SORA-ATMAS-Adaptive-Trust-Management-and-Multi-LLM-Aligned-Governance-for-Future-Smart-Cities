#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 
do
echo "Shahbaz"



#openssl ec -aes-128-cbc -in  private-key$i.pem> /home/ubuntu/Desktop/Shahbaz_Final_Project/Ada-Framework-Shahbaz1/01-GetStarted/Keys/Keys2/keto$i.key


openssl ecparam -name secp128r1 -genkey -noout -out private-key$i.pem > /home/ubuntu/Desktop/Shahbaz_Final_Project/Ada-Framework-Shahbaz1/01-GetStarted/Keys/Keys2/keto$i.key

openssl ec -in private-key$i.pem -pubout -out public-key$i.pem

openssl dgst -sha -binary private-key$i.pem > /home/ubuntu/Desktop/Shahbaz_Final_Project/Ada-Framework-Shahbaz1/01-GetStarted/Keys/Keys2/hash$i.txt
openssl pkeyutl -sign -inkey private-key$i.pem -in hash$i.txt > /home/ubuntu/Desktop/Shahbaz_Final_Project/Ada-Framework-Shahbaz1/01-GetStarted/Keys/Keys2/sig$i.txt

done

file="$1"
echo $file

openssl pkeyutl -verify -in $file -sigfile sig1.txt -inkey private-key1.pem
echo shahbaz
echo $verify
