old.file.names <- unlist(list.files("fps"))      # 返回当前目录下所有文件名集合，类型为向量

for (i in 1:length(old.file.names))
{
  file <- unlist(strsplit(old.file.names[i],split="[.]"))
  new.file.name <- paste0("fp",i,".",file[length(file)])
  # copy to "train"
  file.copy(paste0("fps/",old.file.names[i]),paste0("train/",new.file.name))
}
