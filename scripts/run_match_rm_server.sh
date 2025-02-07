#!/bin/bash

LOC=sh

usage() {
  echo "This script is for PPO-ptx training."
  echo "Options:"
  echo "-l     cephfs location: qy, sh"
}
exit_abnormal() {
  usage
  exit 1
}
while getopts ":l:" options; do
  case "${options}" in
    l)
      LOC=${OPTARG}
      ;;
    :)                                    # If expected argument omitted:
      echo "Error: -${OPTARG} requires an argument."
      exit_abnormal                       # Exit abnormally.
      ;;
    *)                                    # If unknown (any other) option:
      exit_abnormal                       # Exit abnormally.
      ;;
  esac
done

if [ "${LOC}" = "qy" ] ; then
  ROOT_DIR=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/OpenRLHF
elif [ "${LOC}" = "sh" ] ; then
  ROOT_DIR=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat/OpenRLHF
fi

uvicorn --app-dir "${ROOT_DIR}" openrlhf.remote_rm.match_rm_server:app --host 0.0.0.0 --port 1234 >& match_rm.log &
