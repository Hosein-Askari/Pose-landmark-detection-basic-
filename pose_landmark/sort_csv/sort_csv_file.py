import csv


pushups_up=[]
pushups_down=[]
squats_up=[]
squats_down=[]
count1=1
count2=1
count3=1
count4=1
with open('sort_csv/fitness_pose_samples.csv', mode ='r')as file:
  lines = csv.reader(file)
  for line in lines:        
        
        if line[1]=='pushups_up':
             line[0]=f"sample_{count1:0>5}"
             pushups_up.append(line) 
             count1 += 1

        elif line[1]=='pushups_down':
             line[0]=f"sample_{count2:0>5}"
             pushups_down.append(line) 
             count2 += 1

        elif line[1]=='squats_up':
             line[0]=f"sample_{count3:0>5}"
             squats_up.append(line) 
             count3 += 1

        elif line[1]=='squats_down':
             line[0]=f"sample_{count4:0>5}"
             squats_down.append(line) 
             count4 += 1        
        

for i in range (len(pushups_up)):
        with open('sort_csv/pushups_up.csv', mode ='a',newline="")as file:
                writer = csv.writer(file) 
                writer.writerows([pushups_up[i]]) 
file.close() 

for i in range (len(pushups_down)):
        with open('sort_csv/pushups_down.csv', mode ='a',newline="")as file:
                writer = csv.writer(file) 
                writer.writerows([pushups_down[i]]) 
file.close()

for i in range (len(squats_up)):
        with open('sort_csv/squats_up.csv', mode ='a',newline="")as file:
                writer = csv.writer(file) 
                writer.writerows([squats_up[i]]) 
file.close()

for i in range (len(squats_down)):
        with open('sort_csv/squats_down.csv', mode ='a',newline="")as file:
                writer = csv.writer(file) 
                writer.writerows([squats_down[i]]) 
file.close()