trigger ContentDocumentLinkTrigger on ContentDocumentLink (after insert) {
    try{
        Set<String> ruleIDs = new Set<String>();
        List<Messaging.SingleEmailMessage> emails = new List<Messaging.SingleEmailMessage>();
        
        Set<String> documentIDs = new Set<String>();
        Map<String, String> documentLEMap = new Map<String, String>();
        Map<String, String> leTitleMap = new Map<String, String>();
        Map<String, String> leIdMap = new Map<String, String>();
        for(ContentDocumentLink cd: Trigger.New){
            if(((String)cd.LinkedEntityId).startsWith('a05')){
                documentIDs.add(cd.ContentDocumentId);
                ruleIDs.add(cd.LinkedEntityId);
                if(documentLEMap.containsKey(cd.ContentDocumentId)){
	                documentLEMap.put(cd.ContentDocumentId, documentLEMap.get(cd.ContentDocumentId)+cd.LinkedEntityId+';');
                }else{
                	documentLEMap.put(cd.ContentDocumentId, cd.LinkedEntityId+';');                    
                }
            }
        }
        
        for(ContentDocument cd: [SELECT Id, Title, FileType FROM ContentDocument WHERE Id IN: documentIDs]){
            String s = documentLEMap.get(cd.Id);
            List<String> ls = s.split(';');
            for(String l: ls){
	            leTitleMap.put(l, cd.Title+'.'+cd.FileType.toLowerCase());
	            leIdMap.put(l, cd.Id);
            }
        }
        
        for(DFA_Rule__c df: [SELECT Id, Name, Alert_Owner__c, Description__c, Alert_Owner_Name__c, Alert_Owner_Email__c FROM DFA_Rule__c WHERE Id IN: ruleIDs]){
            Messaging.SingleEmailMessage email = new Messaging.SingleEmailMessage();
            
            // Set the recipient email address
            email.setToAddresses(new String[] { df.Alert_Owner_Email__c });
            
            // Set the subject and body dynamically
            email.setSubject('Notification regarding DFA Triggered: ' + df.Name);
            email.setHtmlBody(
                '<p>Dear ' + df.Alert_Owner_Name__c + ',</p>' +
                '<p>We wanted to notify you about the following activity in file <b>"'+leTitleMap.get(df.Id)+'"</b> at <b>'+System.now()+'</b>:</p>' +
                '<p>' + df.Description__c+'</p>' +
                '<p> Access the file here: '+ Url.getOrgDomainUrl().toExternalForm() + '/' + leIdMap.get(df.Id)+'</p>'+
                '<p>Thank you,<br>The Team</p>'
            );
            
            // Prevent logging an email activity record
            email.setSaveAsActivity(true);
            
            emails.add(email);
            
        }
        
        // Send the emails
        if (!emails.isEmpty()) {
            Messaging.sendEmail(emails);
        }        
    }catch(Exception E){
        System.debug('Error '+E.getMessage()+' '+E.getStackTraceString());
    }
}