CREATE TABLE TVIV(
    ID_VIV DECIMAL,
    UPM INTEGER NOT NULL,
    VIV_SEL INTEGER NOT NULL,
    CVE_ENT INTEGER,
    NOM_ENT VARCHAR,
    CVE_MUN INTEGER,
    NOM_MUN VARCHAR,
    COD_RES INTEGER,
    P1_1 INTEGER,
    P1_2 INTEGER,
    P1_2_A INTEGER,
    P1_3 INTEGER,
    P1_4_1 INTEGER,
    P1_4_2 INTEGER,
    P1_4_3 INTEGER,
    P1_4_4 INTEGER,
    P1_4_5 INTEGER,
    P1_4_6 INTEGER,
    P1_4_7 INTEGER,
    P1_4_8 INTEGER,
    P1_4_9 INTEGER,
    P1_5 INTEGER,
    P1_6 INTEGER,
    P1_7 INTEGER,
    P1_8 INTEGER,
    P1_9 DECIMAL,
    P1_10_1 DECIMAL,
    P1_10_2 DECIMAL,
    P1_10_3 DECIMAL,
    P1_10_4 DECIMAL,
    FAC_VIV INTEGER,
    DOMINIO VARCHAR,
    EST_DIS INTEGER,    
    UPM_DIS INTEGER,
    ESTRATO INTEGER,
	PRIMARY KEY (UPM, VIV_SEL)
);
	
CREATE TABLE TSDem (
    ID_VIV DECIMAL,
    ID_PER VARCHAR,
    UPM INTEGER,
    VIV_SEL INTEGER,
    CVE_ENT INTEGER,
    NOM_ENT VARCHAR,
    CVE_MUN INTEGER,
    NOM_MUN VARCHAR,
    HOGAR INTEGER NOT NULL,
    N_REN INTEGER NOT NULL,
    NOMBRE VARCHAR,
    PAREN INTEGER,
    SEXO INTEGER,
    EDAD INTEGER,
    P2_5 INTEGER,
    P2_6 INTEGER,
    NIV DECIMAL,
    GRA DECIMAL,
    P2_8 DECIMAL,
    P2_9 DECIMAL,
    P2_10 DECIMAL,
    P2_11 DECIMAL,
    P2_12 DECIMAL,
    P2_13 DECIMAL,
    P2_14 DECIMAL,
    P2_15 DECIMAL,
    P2_16 DECIMAL,
    COD_M15 DECIMAL,
    CODIGO DECIMAL,
    REN_MUJ_EL DECIMAL,
    REN_INF_AD DECIMAL,
    FAC_VIV INTEGER,
    FAC_MUJ INTEGER,
    DOMINIO VARCHAR,
    ESTRATO INTEGER,
    EST_DIS INTEGER,
    UPM_DIS INTEGER,
	PRIMARY KEY (HOGAR, N_REN, UPM, VIV_SEL),
	FOREIGN KEY (UPM, VIV_SEL) REFERENCES TVIV (UPM, VIV_SEL)
);

CREATE TABLE tb_sec_III(
    ID_VIV DECIMAL ,
    ID_PER VARCHAR ,
    UPM INTEGER ,
    VIV_SEL INTEGER ,
    CVE_ENT INTEGER ,
    NOM_ENT VARCHAR ,
    CVE_MUN INTEGER ,
    NOM_MUN VARCHAR ,
    HOGAR INTEGER ,
    T_INSTRUM VARCHAR ,
    N_REN INTEGER ,
    P3_1 INTEGER ,
    P3_2 DECIMAL ,
    P3_3 DECIMAL ,
    P3_4 DECIMAL ,
    P3_5 DECIMAL ,
    P3_6 DECIMAL ,
    P3_7 DECIMAL ,
    P3_8 VARCHAR ,
    FAC_VIV INTEGER ,
    FAC_MUJ INTEGER ,
    DOMINIO VARCHAR ,
    ESTRATO INTEGER ,
    EST_DIS INTEGER ,
    UPM_DIS INTEGER ,
	PRIMARY KEY (HOGAR, N_REN, UPM, VIV_SEL),
	FOREIGN KEY (UPM, VIV_SEL) REFERENCES TVIV (UPM, VIV_SEL)
);

CREATE TABLE tb_sec_IV(
    ID_VIV DECIMAL ,
    ID_PER VARCHAR ,
    UPM INTEGER ,
    VIV_SEL INTEGER ,
    HOGAR INTEGER ,
    N_REN INTEGER ,
    DOMINIO VARCHAR ,
    CVE_ENT INTEGER ,
    NOM_ENT VARCHAR ,
    CVE_MUN INTEGER ,
    NOM_MUN VARCHAR ,
    T_INSTRUM VARCHAR ,
    N_REN_ESP DECIMAL ,
    P4AB_1 DECIMAL ,
    P4AB_2 DECIMAL ,
    P4A_1 DECIMAL ,
    P4A_2 DECIMAL ,
    P4B_1 DECIMAL ,
    P4B_2 DECIMAL ,
    P4BC_1 DECIMAL ,
    P4BC_2 DECIMAL ,
    P4C_1 DECIMAL ,
    P4BC_3 DECIMAL ,
    P4BC_4 DECIMAL ,
    P4BC_5 DECIMAL ,
    P4_1 INTEGER ,
    P4_2 DECIMAL ,
    P4_2_1 DECIMAL ,
    P4_3 DECIMAL ,
    P4_4 VARCHAR ,
    P4_4_CVE DECIMAL ,
    P4_5_AB DECIMAL ,
    P4_5_1_AB DECIMAL ,
    P4_6_AB DECIMAL ,
    P4_7_AB DECIMAL ,
    P4_8_1 INTEGER ,
    P4_8_2 INTEGER ,
    P4_8_3 INTEGER ,
    P4_8_4 INTEGER ,
    P4_8_5 INTEGER ,
    P4_8_6 INTEGER ,
    P4_8_7 INTEGER ,
    P4_9_1 DECIMAL ,
    P4_9_2 DECIMAL ,
    P4_10_2_1 DECIMAL ,
    P4_10_2_2 DECIMAL ,
    P4_10_2_3 DECIMAL ,
    P4_9_3 DECIMAL ,
    P4_10_3_1 DECIMAL ,
    P4_10_3_2 DECIMAL ,
    P4_10_3_3 DECIMAL ,
    P4_9_4 DECIMAL ,
    P4_9_5 DECIMAL ,
    P4_9_6 DECIMAL ,
    P4_9_7 DECIMAL ,
    P4_11 INTEGER ,
    P4_12_1 INTEGER ,
    P4_12_2 INTEGER ,
    P4_12_3 INTEGER ,
    P4_12_4 INTEGER ,
    P4_12_5 INTEGER ,
    P4_12_6 INTEGER ,
    P4_12_7 INTEGER ,
    P4_13_1 DECIMAL ,
    P4_13_2 DECIMAL ,
    P4_13_3 DECIMAL ,
    P4_13_4 DECIMAL ,
    P4_13_5 DECIMAL ,
    P4_13_6 DECIMAL ,
    P4_13_7 DECIMAL ,
    FAC_VIV INTEGER ,
    FAC_MUJ INTEGER ,
    ESTRATO INTEGER ,
    UPM_DIS INTEGER ,
    EST_DIS INTEGER ,
	PRIMARY KEY (HOGAR, N_REN, UPM, VIV_SEL),
	FOREIGN KEY (UPM, VIV_SEL) REFERENCES TVIV (UPM, VIV_SEL)
);

CREATE TABLE tb_sec_x(
    ID_VIV DECIMAL,
    ID_PER VARCHAR,
    UPM INTEGER NOT NULL,
    VIV_SEL INTEGER NOT NULL,
    HOGAR INTEGER NOT NULL,
    N_REN INTEGER NOT NULL,
    DOMINIO VARCHAR,
    CVE_ENT INTEGER,
    NOM_ENT VARCHAR,
    CVE_MUN INTEGER,
    NOM_MUN VARCHAR,
    T_INSTRUM VARCHAR,
    P10_1_1 DECIMAL,
    P10_1_2 DECIMAL,
    P10_1_3 DECIMAL,
    P10_1_4 DECIMAL,
    P10_1_5 DECIMAL,
    P10_1_6 DECIMAL,
    P10_1_7 DECIMAL,
    P10_1_8 DECIMAL,
    P10_1_9 DECIMAL,
    P10_2 DECIMAL,
    P10_3 DECIMAL,
    P10_4_1 DECIMAL,
    P10_4_2 DECIMAL,
    P10_4_3 DECIMAL,
    P10_5_01 DECIMAL,
    P10_5_02 DECIMAL,
    P10_5_03 DECIMAL,
    P10_5_04 DECIMAL,
    P10_5_05 DECIMAL,
    P10_5_06 DECIMAL,
    P10_5_07 DECIMAL,
    P10_5_08 DECIMAL,
    P10_5_09 DECIMAL,
    P10_5_10 DECIMAL,
    P10_5_11 DECIMAL,
    P10_6ANIO DECIMAL,
    P10_6MES DECIMAL,
    P10_7 DECIMAL,
    P10_8_1 DECIMAL,
    P10_8_2 DECIMAL,
    P10_8_3 DECIMAL,
    P10_8_4 DECIMAL,
    P10_8_5 DECIMAL,
    P10_8_6 DECIMAL,
    P10_8_7 DECIMAL,
    P10_8_8 DECIMAL,
    P10_8_9 DECIMAL,
    P10_8_10 DECIMAL,
    P10_8_11 DECIMAL,
    P10_8_12 DECIMAL,
    P10_8_13 DECIMAL,
    P10_8_14 DECIMAL,
    P10_8_15 DECIMAL,
    FAC_VIV INTEGER,
    FAC_MUJ INTEGER,
    ESTRATO INTEGER,
    UPM_DIS INTEGER,
    EST_DIS INTEGER,
	PRIMARY KEY (HOGAR, N_REN, UPM, VIV_SEL),
	FOREIGN KEY (UPM, VIV_SEL) REFERENCES TVIV (UPM, VIV_SEL)
);