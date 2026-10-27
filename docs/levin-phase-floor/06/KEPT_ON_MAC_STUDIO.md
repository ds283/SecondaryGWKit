# Files of prompt 06 kept on the Mac Studio only

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06/` (paths below are relative
to it). They were left out of this directory for size, as 05's and 05a's census CSVs and large
outputs were: every pickle, every file over 2 MB, every CSV, the value comparison's JSON, the
readers' extracts and the scratch copy of A3 v2. The run directories under `var/runs/`
(`levin-phase-floor-06-r13-parent-20261026T215550`, `…-r14-residual-…`,
`levin-phase-floor-06-wider-20261026T223634`) and the R13/R14 stores stay there as well.

The SHA-256 of each was taken on the Mac Studio on 2026-10-27, when the records were copied back.
The scratch copy of A3 v2 is certified by `wider/state/copy_files_certified.txt` (fingerprint
`0e32f24a…`); it is kept until the user says otherwise.

```
2d42f77114f4a3dfa3176fb9fc241c26db84239134ae6c0f94c2e2e00200239a  ./p1/out/analyse_r13_pieces.npz
1690b553412a55e256f7215780912891254f359ac54547041b85162a5116cdc4  ./p1/out/analyse_r14_pieces.npz
5f44c662efabd93870ad955a9225208098f2cf3a1f6f146192afb8b5d3255b8f  ./p1/out/samples_r13.npz
c8aa03ea8d3eebd282a609264af2b9f26cb6aa29d1a460406fb438ca0f1c431d  ./p1/out/samples_r14.npz
89b5ddb62c4b799596fa0409f1a48116ce1564025c79444aa438c78be447cd5e  ./rehearsal/out/phase_classes_rows.csv
88c8045ff718e523b4ceeb092cf23832817e2a765e8384932408649f7135da39  ./rehearsal/out/qsi_rows.csv
7e42eadcf3fc6a33b7ced1eed0dca813236d6e189e5aee051aa989bc644069de  ./rehearsal/out/store_value_diff.json
27fb674a1eb15c46611301e4f54e24f0241781474b2896958cbf048e585d48bc  ./wider/state/U0/s/ledger.jsonl
d0d7c2626c7ae0fda889fc6fb04a205692bb385bcea8caefefa5f327301e316a  ./wider/state/U0/s/unit_U0-s-1e-7-ctrl-low-1-run1.pkl
de8d3b6c8f35b8b889821a853feb712fcc88cb41f1f9363a83f317e1506b73c4  ./wider/state/U0/s/unit_U0-s-1e-7-ctrl-low-1-run2.pkl
d35ce50800d3f6bc53f1d5846c28a4d79442dd9595cf3253b4433270581428e6  ./wider/state/U0/s/unit_U0-s-1e-7-low-G+r-1-run1.pkl
04bb1dbb97a73833f0f1942b0312eac78ac08d0f3a1ca53fc88b09a6795ba4c7  ./wider/state/U0/s/unit_U0-s-1e-7-low-G+r-1-run2.pkl
e5aeaad15367a2a8fc0bf2e7b72031e3618ee948508f874370f8773666c6f55c  ./wider/state/U0/t5/ledger.jsonl
3c2d14a6da3f7c99e652c7363841472a59de83098238965e0fde067f16988ef4  ./wider/state/U0/t5/unit_U0-t5-1e-7-ctrl-low-1.pkl
53241de9985e69a6798c11dfd6fc13e865d8c7fa4dcb0d8a87e753e1cad8cad0  ./wider/state/U0/t5/unit_U0-t5-1e-7-low-G+r-1.pkl
3565feeae5d2e0c1815e9ed18eb28406d4ca263b21a549d1306eb55f84bcbca0  ./wider/state/U0/t5/unit_U0-t5-rerun-1e-7-ctrl-low-1.pkl
8cb02c70fa88f0bd266d498d4f8a20c988cface48520f1771ec653d85c1aa7b4  ./wider/state/U0/t5/unit_U0-t5-rerun-1e-7-low-G+r-1.pkl
ba72e3502075f28dd60baf012f7d4c390e76b90d475b5e74f918e796a675bcbc  ./wider/state/dev/s2/ledger.jsonl
eb6092b188e1f9fd6eff6b65f054230efc95de020372c1bf0886fbad293ce3bf  ./wider/state/dev/s2/unit_dev2-s-1e-7-ctrl-low-1.pkl
6a680b089749db1b4020bc42e5a25091d153f59ec7d0962ea213ad9a48d4dc80  ./wider/state/dev/t5/ledger.jsonl
99ef18e03aa160b55e988047eed6d5e4a1fabd25be93c43680014be97227b6da  ./wider/state/dev/t5/unit_dev-t5-1e-7-ctrl-low-1.pkl
a81645a62a22257befd27842815ce1ee2edf2177ede9bfb84a16a227721c60a4  ./wider/state/probe_windows.json
7b4d2d135453355668953d60919a7ab981993a5a3f02e0ab5676f97c092ec517  ./wider/state/smoke/ledger.jsonl
53e3e3792ec4e44aebcda5d849bf60cca54c5e80e1604ca1f403ae4665574e4e  ./wider/state/smoke/unit_s-1e-7-R7.pkl
50d67737e6c38ee5d8a196350ecc74ea3372732f7a6241407c8e6337108519f5  ./wider/state/smoke/unit_s-1e-7-ctrl-NULL-1.pkl
05b0927338d68ae3a379af0aa520e1378cfb1e85b61d46918543bdd002487319  ./wider/state/smoke/unit_s-1e-8-R7.pkl
9127e4d5d1afd6d94d65b55bb014b2543d08f64f4163bac7c09120186810c78e  ./wider/state/smoke/unit_t5-1e-7-R7.pkl
1a759e0d6fc6c632b0abeef64e395452b258b0fdd7c950e6ed1f05ecc790a578  ./wider/state/smoke/unit_t5-1e-7-ctrl-NULL-1.pkl
# rehearsal/extract:     3138 files, 176M
# wider/state/a3v2_copy:
-rw-r--r--  1 ds283  staff  106340352 Oct 26 22:19 handover-A3-v2-lambdacdm-shard0000.sqlite
-rw-r--r--  1 ds283  staff  107569152 Oct 26 22:19 handover-A3-v2-lambdacdm-shard0001.sqlite
-rw-r--r--  1 ds283  staff  107429888 Oct 26 22:19 handover-A3-v2-lambdacdm-shard0002.sqlite
-rw-r--r--  1 ds283  staff  105791488 Oct 26 22:19 handover-A3-v2-lambdacdm-shard0003.sqlite
-rw-r--r--  1 ds283  staff      36864 Oct 26 22:19 handover-A3-v2-lambdacdm.sqlite
```
