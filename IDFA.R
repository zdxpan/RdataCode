library(jsonlite)
library(openxlsx);
library(dplyr)
beday = Sys.Date() - 3
today = Sys.Date()
today = as.character(today)
beday = as.character(beday)

ur1 = "http://console.mobile.dianru.com/ads_ios_api/callback.do?&actions=4&ad_from=&adid="
ur2 = "&invalid=&mac=&page=1&process=&saved=&time=1498612777&udid=&uid=&secret=e8e766d90edafc7939e230d5d54c3e1e"
outx2 = function(t,ta = tar,ad = adname,where = "2017-07-28-30"){
  wb <- createWorkbook("Fred")
  t$date = as.character(t$date)
  for(i in as.character(unique( as.Date(t$date)))  )
  {
    #i = unique(t$date)[1]
    temp =   t[which(as.Date.character(t$date)==i),]
    #temp$渠道 = "点入"
    sheet = i
    sheet = stringi::stri_replace_all(sheet,replacement = "-",regex = "/")
    
    sheet = paste(sheet,"-",sep="")
    sheet = paste(sheet,as.character(nrow(temp)),sep="")
    addWorksheet(wb, sheet)
    writeData(wb, sheet = sheet, temp)
  }
  
  saveWorkbook(wb, paste(ta,where,ad,".xlsx",sep = ""), overwrite = TRUE)
  rm(i,sheet,temp,wb)
}
outex1 = function(t)
{ 
  nm = as.Date.character(t[1,"date"])
  nm = paste(nm,adname,nrow(t),sep = "-")
  nm = paste(nm,".xlsx",sep = "")
  nm = paste(tar,nm,sep = "")
  #write.csv(t,tar)
  openxlsx::write.xlsx(x = t,file = nm)#addWorksheet(createWorkbook("Fred"),sheetName = "zdx",gridLines = T,tabColour = "red"),asTable = T)
  rm(t,nm)
}

loaddr = function(u1 = ur1,ad = adid,bday = beday,tday = today,u2 = ur2)
{ 
  library(jsonlite)
  library(dplyr)
  temp = data.frame(NULL)
  date = as.character(seq(bday,as.Date(tday),by = 'day'))
  for(j in  date){
    print(j)
    url = paste(u1,ad,"&appid=&beginDate=",j,"&cid=&data_from=&dbtype=1&endDate=",j,u2,sep = "")
    order =  fromJSON(url)#JSON string contains (illegal) UTF8 byte-order-mark! 
    order = order$list
    order = dplyr::select(order,udid,create_time_title,open_time)
    names(order) = c("idfa","date","time")
    
    order = order %>%
      mutate(date = paste(date,time)) %>%
      select(idfa,date)
   # order = order[]
    temp = rbind(temp,order)
    #return(order)
  }
  return(temp)
}

vf = function(td = today,ad = adid)
{
    tod = paste(substr(td,1,4),substr(td,6,7),substr(td,9,10),sep = "")
    ur = "http://console.vfou.com/api/aidInfo?date="  
    ur = paste(ur,tod,"&adid=",ad,sep = "")
    order = jsonlite::fromJSON(ur)
    if(!is.null(nrow(order) ) ){
      names(order) = c("idfa","date") 
    }else{
      return(NULL)
    }
    return(order)
}

vfa = function(ad = adid,bday = beday,tday = today){
  
  temp = data.frame(NULL)
  date = as.character(seq(bday,as.Date(tday),by = 'day'))
  for(j in  date){
    print(j)
    v1 = vf(td = j,ad = ad)
    temp = rbind(temp,v1)
  }
  return(temp)
}

te = read.delim("clipboard", header =T, sep = "\t")#区分是哪天的数据
te$激活时间  = chartr(te$激活时间,"/","-")
te$激活时间 = gsub( '/', '-',te$激活时间)
names(te) = c("idfa","date")




t = splitstackshape::cSplit(t,splitCols = 'date',sep = ' ')
names(t) = c("idfa","date","time")
summarise(group_by(t,date),n())
unique(as.Date(t$date))

vf2 = function(ifsep){
  t = read.table("clipboard", header =T, sep = ",")#区分是哪天的数据
  t = t[,c("日期","IDFA")]
  names(t) = c("date","idfa")
  if(ifsep){
    t = splitstackshape::cSplit(t, 'date', sep=" ", type.convert=FALSE)
    names(t) = c("idfa","date","time")
  }
  return(t)
}

tar = "E:/1workdianru/1李硕/"
adname = "更美";adid = 15611;t = loaddr(ad = adid);outex1(t);#outx2(t = t,where = "2017-08-11-13")#

adname = "借钱花-李硕";adid = 18333;te = loaddr(ad = adid);t = vfa();t= rbind(t,te);outex1(t)

adname = "现金贷";adid = 18352;te = loaddr(ad = adid);t = vf();t= rbind(t,te);outex1(t)


tar = "E:/1workdianru/2张俊/"
adname = "优信二手车";adid = 16147;t = loaddr(ad = adid);t$渠道 = "点入";outex1(t)#outx2(t = t,where = "2017-04-07")
adname = "手机百度-张俊";adid = 18298


adname = "金储宝";adid = 18391;te = loaddr(ad = adid);t = vfa();t= rbind(t,te);outex1(t)
t = loaddr(ad = adid)

adname ="三五彩票-秦少楠-接";adid = 18388

adname = '触手';adid = 18386


te = loaddr(ad = adid)
t = vf()
t= rbind(t,te)

outex1(t)

outx2(t)

loaddr2 = function(u1 = ur1,ad = adid,bday = beday,tday = today,u2 = ur2)
{ 
  
  url = paste(u1,ad,"&appid=&beginDate=",bday,"&cid=&data_from=&dbtype=1&endDate=",tday,u2,sep = "")
  print(url)
  order =  fromJSON(url)#JSON string contains (illegal) UTF8 byte-order-mark! 
  order = order$list
  
  return(order)
}
netnew = read.csv("clipboard",header = F,sep = ',')
str = gsub("([\\])","", dat$V1)
stringr::str_replace_all()
names(dat)
dat$date = as.character(dat$V1)
dat$date = as.Date(dat$date)


vf_idfa = te[1,]
vf_idfa = vf_idfa[-1,]
datv = read.csv("clipboard",header = F,sep = '\t')
names(datv) = c("date","num")
datv$date = as.Date(datv$date)
datv$date = as.character(datv$date)

for(i in datv$date){
  #te = loaddr(ad = adid,bday = i,tday = i)
  #vf = function(td = today,ad = adid)
  
  datavf = vf(td = i,ad = adid);
  #t= rbind(t,te)
  vf_idfa = rbind(vf_idfa,datavf)
}

vf_xpa = temp

dat = left_join(dat,datv,by = "date")
dat[is.na(dat$num.y),'num.y']=0
dat$dr = dat$num.x-dat$num.y

temp = te[1,]
temp = temp[-1,]
dat = seq.Date(from = as.Date('2017-07-01'),to = as.Date('2017-07-21'),by = "day")
dat = as.character(dat)
for(i in dat){
  te = loaddr(ad = adid,bday = i,tday = i)
  temp = rbind(temp,te)
}
temp = rbind(temp,vf_xpa)
dr_xpa = temp
temp = splitstackshape::cSplit(temp,splitCols = 'date',sep = ' ',type.convert = F)
names(temp) = c("idfa","date","time")
unique(temp$date)
summarise(group_by(temp,date),n())

openxlsx::write.xlsx(x = temp,file = "E:/1workdianru/李硕/2017-07-01-21更美-李朔.xlsx")

outx2 = function(t,ta = tar,ad = adname){
  wb <- createWorkbook("Fred")
  t$date = as.character(t$date)
  for(i in as.character(unique( as.Date(t$date)))  )
  {
    #i = unique(t$date)[1]
    temp =   t[which(as.Date.character(t$date)==i),]
    #temp$渠道 = "点入"
    sheet = i
    sheet = stringi::stri_replace_all(sheet,replacement = "-",regex = "/")
    
    sheet = paste(sheet,"-",sep="")
    sheet = paste(sheet,as.character(nrow(temp)),sep="")
    addWorksheet(wb, sheet)
    writeData(wb, sheet = sheet, temp)
  }
  
  saveWorkbook(wb, paste(ta,"2017-07-28-30",ad,".xlsx",sep = ""), overwrite = TRUE)
  rm(i,sheet,temp,wb)
}

t[1,]
outx2 = function(t,ta = tar,ad = adname,where = "2017-07-28-30"){
  wb <- createWorkbook("Fred")
  t$date = as.character(t$date)
  for(i in as.character(unique( as.Date(t$date)))  )
  {
    #i = unique(t$date)[1]
    temp =   t[which(as.Date.character(t$date)==i),]
    #temp$渠道 = "点入"
    sheet = i
    sheet = stringi::stri_replace_all(sheet,replacement = "-",regex = "/")
    
    sheet = paste(sheet,"-",sep="")
    sheet = paste(sheet,as.character(nrow(temp)),sep="")
    addWorksheet(wb, sheet)
    writeData(wb, sheet = sheet, temp)
  }
  
  saveWorkbook(wb, paste(ta,where,ad,".xlsx",sep = ""), overwrite = TRUE)
  rm(i,sheet,temp,wb)
}

cat("Mean=", 45, "\n", "SD=", 467, "\n")

x = read.csv('clipboard',header = T,sep = '\t')
summarise(group_by(x,渠道名称),sum(核减量级))
